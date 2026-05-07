import wx
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.patches as patches
import matplotlib.lines as lines
import numpy as np
import math
import os
import json
import subprocess
import re
import gmsh

# --- Helper Functions ---
def point_segment_distance_sq_t(px, py, x1, y1, x2, y2):
    line_vec_x, line_vec_y = x2 - x1, y2 - y1
    point_vec_x, point_vec_y = px - x1, py - y1
    line_len_sq = line_vec_x**2 + line_vec_y**2
    if line_len_sq < 1e-12:
        return point_vec_x**2 + point_vec_y**2, 0.0
    t = (point_vec_x * line_vec_x + point_vec_y * line_vec_y) / line_len_sq
    t_clamped = max(0.0, min(1.0, t))
    closest_x = x1 + t_clamped * line_vec_x
    closest_y = y1 + t_clamped * line_vec_y
    dist_sq = (px - closest_x)**2 + (py - closest_y)**2
    return dist_sq, t_clamped

class PointLineEditorPanel(wx.Panel):
    def __init__(self, parent, statusbar=None):
        super().__init__(parent)
        self.parent_frame = wx.GetTopLevelParent(self)
        self.statusbar = statusbar
        
        self.points = []
        self.markers = []
        self.segments = []
        
        self.selected_point_index = None
        self.selected_marker = None
        self.selected_segment_index = None
        self.selected_segment_artist = None
        self.mode = "add_init_points"
        self.loop_closed = False
        self.dragging_point = False
        self.is_dirty = False
        self.current_project_path = None
        
        self.init_ui()
        self.update_status_bar()

    def init_ui(self):
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlim(0, 100)
        self.axes.set_ylim(0, 100)
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.grid(True)
        self.canvas = FigureCanvas(self, -1, self.figure)
        
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 0)
        sizer.Add(self.toolbar, 0, wx.EXPAND | wx.ALL, 0)
        self.SetSizer(sizer)

    def notify_selection_changed(self):
        if hasattr(self.parent_frame, "on_selection_changed"):
            self.parent_frame.on_selection_changed()

    def mark_dirty(self):
        self.is_dirty = True
        if hasattr(self.parent_frame, "on_editor_changed"):
            self.parent_frame.on_editor_changed()

    def set_mode(self, new_mode):
        if self.mode == new_mode: return
        self.mode = new_mode
        self.deselect_point()
        self.deselect_segment()
        self.update_status_bar()
        self.canvas.draw_idle()
        self.notify_selection_changed()

    def deselect_point(self):
        if self.selected_marker:
            self.selected_marker.set_color('red')
            self.selected_marker.set_markersize(5)
            self.selected_marker = None
        self.selected_point_index = None
        self.update_status_bar()

    def deselect_segment(self):
        if self.selected_segment_artist:
            artist = self.selected_segment_artist
            if isinstance(artist, lines.Line2D):
                artist.set_color('blue')
                artist.set_linewidth(1.0)
            elif isinstance(artist, patches.Arc):
                artist.set_color('green')
                artist.set_linewidth(1.0)
            self.selected_segment_artist = None
        self.selected_segment_index = None
        self.update_status_bar()

    def on_click(self, event):
        if event.inaxes != self.axes:
            self.deselect_point()
            self.deselect_segment()
            self.notify_selection_changed()
            return
        
        x, y = event.xdata, event.ydata
        px, py = event.x, event.y

        if self.mode == "add_init_points":
            if not self.loop_closed:
                idx = len(self.points)
                self.points.append((x, y))
                self.markers.append(self.axes.plot(x, y, 'ro', markersize=5, picker=5)[0])
                if idx >= 1:
                    p1, p2 = self.points[idx-1], self.points[idx]
                    l = self.axes.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')[0]
                    self.segments.append({'type': 'line', 'points': [idx-1, idx], 'artist': l, 'physical_name': 'PEC'})
                self.mark_dirty()
                self.canvas.draw()
            else:
                self.select_point(px, py)
        elif self.mode == "edit_points":
            if self.select_point(px, py):
                self.dragging_point = True
            elif event.dblclick:
                self.add_point_on_segment(x, y)
        elif self.mode == "edit_lines":
            self.select_segment(x, y)
        self.notify_selection_changed()

    def select_point(self, px, py):
        for i, m in enumerate(self.markers):
            mx, my = m.get_xdata()[0], m.get_ydata()[0]
            mpx, mpy = self.axes.transData.transform((mx, my))
            if (px - mpx)**2 + (py - mpy)**2 < 100:
                self.deselect_point()
                self.selected_point_index, self.selected_marker = i, m
                m.set_color('green')
                m.set_markersize(8)
                self.canvas.draw_idle()
                self.update_status_bar()
                return True
        self.deselect_point()
        return False

    def select_segment(self, x, y):
        min_d_sq, found_idx = float('inf'), -1
        xlim, ylim = self.axes.get_xlim(), self.axes.get_ylim()
        th_sq = ((xlim[1] - xlim[0]) * 0.05)**2
        for i, seg in enumerate(self.segments):
            d_sq = float('inf')
            if seg['type'] == 'line':
                xd, yd = seg['artist'].get_data()
                d_sq, _ = point_segment_distance_sq_t(x, y, xd[0], yd[0], xd[1], yd[1])
            elif seg['type'] == 'arc':
                c, r = seg['center'], seg['radius']
                angle = math.degrees(math.atan2(y - c[1], x - c[0]))
                
                t1, t2 = seg.get('theta1', 0), seg.get('theta2', 360)
                
                # Normalize angle to the range [t1, t1 + 360)
                while angle < t1: angle += 360
                while angle >= t1 + 360: angle -= 360
                
                if t1 <= angle <= t2:
                    d_sq = (math.sqrt((x-c[0])**2 + (y-c[1])**2) - r)**2
                else:
                    # If outside the drawn arc, distance is the min to endpoints
                    p1_idx, p2_idx = seg['points'][0], seg['points'][1]
                    p1, p2 = self.points[p1_idx], self.points[p2_idx]
                    d1_sq = (x - p1[0])**2 + (y - p1[1])**2
                    d2_sq = (x - p2[0])**2 + (y - p2[1])**2
                    d_sq = min(d1_sq, d2_sq)
            if d_sq < min_d_sq:
                min_d_sq, found_idx = d_sq, i
        if found_idx != -1 and min_d_sq < th_sq:
            self.deselect_segment()
            self.selected_segment_index = found_idx
            self.selected_segment_artist = self.segments[found_idx]['artist']
            color = 'magenta' if self.segments[found_idx]['type'] == 'line' else 'orange'
            self.selected_segment_artist.set_color(color)
            self.selected_segment_artist.set_linewidth(2.0)
            self.canvas.draw_idle()
            self.update_status_bar()
            return True
        self.deselect_segment()
        return False

    def add_point_on_segment(self, x, y):
        min_d_sq, c_idx, bt = float('inf'), -1, 0
        for i, seg in enumerate(self.segments):
            if seg['type'] == 'line':
                p1, p2 = self.points[seg['points'][0]], self.points[seg['points'][1]]
                d_sq, t = point_segment_distance_sq_t(x, y, p1[0], p1[1], p2[0], p2[1])
                if d_sq < min_d_sq:
                    min_d_sq, c_idx, bt = d_sq, i, t
        xlim, ylim = self.axes.get_xlim(), self.axes.get_ylim()
        if c_idx != -1 and min_d_sq < ((xlim[1]-xlim[0])*0.05)**2:
            seg = self.segments[c_idx]
            p1_i, p2_i = seg['points']
            p1, p2 = self.points[p1_i], self.points[p2_i]
            nx, ny = p1[0]+bt*(p2[0]-p1[0]), p1[1]+bt*(p2[1]-p1[1])
            new_i = len(self.points)
            self.points.append((nx, ny))
            self.markers.append(self.axes.plot(nx, ny, 'ro', markersize=5, picker=5)[0])
            pname = seg.get('physical_name', 'PEC')
            seg['artist'].remove()
            self.segments.pop(c_idx)
            l1 = self.axes.plot([p1[0], nx], [p1[1], ny], 'b-')[0]
            self.segments.insert(c_idx, {'type': 'line', 'points': [p1_i, new_i], 'artist': l1, 'physical_name': pname})
            l2 = self.axes.plot([nx, p2[0]], [ny, p2[1]], 'b-')[0]
            self.segments.insert(c_idx+1, {'type': 'line', 'points': [new_i, p2_i], 'artist': l2, 'physical_name': pname})
            self.mark_dirty()
            self.canvas.draw()
            self.notify_selection_changed()

    def on_motion(self, event):
        if not self.dragging_point or self.selected_point_index is None or event.inaxes != self.axes:
            return
        idx, nx, ny = self.selected_point_index, event.xdata, event.ydata
        self.points[idx] = (nx, ny)
        self.selected_marker.set_data([nx], [ny])
        self.update_segments_for_point(idx)
        self.canvas.draw_idle()
        self.notify_selection_changed()

    def on_release(self, event):
        if self.dragging_point:
            self.dragging_point = False
            self.mark_dirty()

    def _convert_arc_to_line(self, idx):
        seg = self.segments[idx]
        if seg['type'] != 'arc': return
        p1, p2 = self.points[seg['points'][0]], self.points[seg['points'][1]]
        seg['artist'].remove()
        l = self.axes.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')[0]
        seg.update({'type': 'line', 'artist': l})
        for k in ['center', 'radius', 'theta1', 'theta2']:
            seg.pop(k, None)
        return l

    def update_segments_for_point(self, pt_idx):
        for i, seg in enumerate(self.segments):
            if pt_idx in seg['points']:
                if seg['type'] == 'arc':
                    self._convert_arc_to_line(i)
                else:
                    p1, p2 = self.points[seg['points'][0]], self.points[seg['points'][1]]
                    seg['artist'].set_data([p1[0], p2[0]], [p1[1], p2[1]])

    def update_status_bar(self):
        if not self.statusbar: return
        msg = f"Mode: {self.mode}"
        if self.selected_point_index is not None:
            msg += f" | Selected Pt: {self.selected_point_index}"
        elif self.selected_segment_index is not None:
            msg += f" | Selected Segment: {self.selected_segment_index}"
        self.statusbar.SetStatusText(msg)

    def close_loop(self):
        if len(self.points) >= 3 and not self.loop_closed:
            p1, p2 = self.points[-1], self.points[0]
            l = self.axes.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')[0]
            self.segments.append({'type': 'line', 'points': [len(self.points)-1, 0], 'artist': l, 'physical_name': 'PEC'})
            self.loop_closed = True
            self.mark_dirty()
            self.canvas.draw()
            return True
        return False

    def reset(self):
        self.points, self.markers, self.segments = [], [], []
        self.selected_point_index, self.selected_segment_index = None, None
        self.loop_closed = False
        self.axes.cla()
        self.axes.grid(True)
        self.canvas.draw()
        self.mark_dirty()

    def update_point_coords(self, idx, x, y):
        if idx < 0 or idx >= len(self.points): return
        self.points[idx] = (x, y)
        self.markers[idx].set_data([x], [y])
        self.update_segments_for_point(idx)
        self.mark_dirty()
        self.canvas.draw_idle()

    def delete_point(self, idx):
        if idx < 0 or idx >= len(self.points): return
        to_rem, prev_pt, next_pt = [], -1, -1
        for i, seg in enumerate(self.segments):
            if idx in seg['points']:
                to_rem.append(i)
                other = seg['points'][0] if seg['points'][1] == idx else seg['points'][1]
                if seg['points'][1] == idx: prev_pt = other
                else: next_pt = other
        for i in sorted(to_rem, reverse=True):
            self.segments.pop(i)['artist'].remove()
        self.markers.pop(idx).remove()
        self.points.pop(idx)
        for seg in self.segments:
            seg['points'] = [p-1 if p>idx else p for p in seg['points']]
        if prev_pt > idx: prev_pt -= 1
        if next_pt > idx: next_pt -= 1
        if prev_pt != -1 and next_pt != -1 and prev_pt != next_pt:
            p1, p2 = self.points[prev_pt], self.points[next_pt]
            l = self.axes.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')[0]
            self.segments.append({'type': 'line', 'points': [prev_pt, next_pt], 'artist': l, 'physical_name': 'PEC'})
        self.deselect_point()
        self.mark_dirty()
        self.canvas.draw()

    def convert_selected_to_arc(self):
        idx = self.selected_segment_index
        if idx is None or self.segments[idx]['type'] != 'line': return
        seg = self.segments[idx]
        p1_i, p2_i = seg['points']
        p1, p2 = np.array(self.points[p1_i]), np.array(self.points[p2_i])
        mid, vec = (p1 + p2) / 2.0, p2 - p1
        dist_p1p2 = np.linalg.norm(vec)
        if dist_p1p2 < 1e-12: return
        perp = np.array([-vec[1], vec[0]])
        center = mid + (perp / dist_p1p2) * (dist_p1p2 / 2.0)
        r = np.linalg.norm(p1 - center)
        a1 = math.degrees(math.atan2(p1[1]-center[1], p1[0]-center[0]))
        a2 = math.degrees(math.atan2(p2[1]-center[1], p2[0]-center[0]))
        
        # --- 短い方の弧(劣弧)を選択するロジック ---
        da = a2 - a1
        while da <= -180: da += 360
        while da > 180: da -= 360
        
        if da >= 0:
            theta1, theta2 = a1, a1 + da
        else:
            theta1, theta2 = a2, a2 + abs(da)
            
        if theta2 <= theta1: theta2 += 360
        # ---------------------------------------

        seg['artist'].remove()
        arc = patches.Arc(tuple(center), width=2*r, height=2*r, angle=0, theta1=theta1, theta2=theta2, color='green', linewidth=1.0)
        self.axes.add_patch(arc)
        seg.update({'type': 'arc', 'center': tuple(center), 'radius': r, 'theta1': theta1, 'theta2': theta2, 'artist': arc})
        self.mark_dirty()
        self.canvas.draw_idle()

    def convert_selected_to_line(self):
        idx = self.selected_segment_index
        if idx is None or self.segments[idx]['type'] != 'arc': return
        seg = self.segments[idx]
        p1, p2 = self.points[seg['points'][0]], self.points[seg['points'][1]]
        seg['artist'].remove()
        l = self.axes.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')[0]
        seg.update({'type': 'line', 'artist': l})
        for k in ['center', 'radius', 'theta1', 'theta2']:
            seg.pop(k, None)
        self.mark_dirty()
        self.canvas.draw_idle()

    def reverse_loop(self):
        if not self.loop_closed or not self.segments: return
        self.segments.reverse()
        for seg in self.segments:
            seg['points'] = [seg['points'][1], seg['points'][0]]
            if seg['type'] == 'arc':
                p1, p2, c = self.points[seg['points'][0]], self.points[seg['points'][1]], seg['center']
                # 逆転後の新しい始点・終点の角度
                a1 = math.degrees(math.atan2(p1[1]-c[1], p1[0]-c[0]))
                a2 = math.degrees(math.atan2(p2[1]-c[1], p2[0]-c[0]))
                
                # 最短経路（180度以内）を維持するための正規化
                da = a2 - a1
                while da <= -180: da += 360
                while da > 180: da -= 360
                
                if da >= 0:
                    theta1, theta2 = a1, a1 + da
                else:
                    theta1, theta2 = a2, a2 + abs(da)
                
                if theta2 <= theta1: theta2 += 360
                
                seg.update({'theta1': theta1, 'theta2': theta2})
                seg['artist'].theta1, seg['artist'].theta2 = theta1, theta2
        self.mark_dirty()
        self.canvas.draw_idle()

    def update_arc_center(self, idx, cx, cy):
        if idx < 0 or idx >= len(self.segments): return
        seg = self.segments[idx]
        if seg['type'] != 'arc': return
        p1, p2, center = np.array(self.points[seg['points'][0]]), np.array(self.points[seg['points'][1]]), np.array([cx, cy])
        r = np.linalg.norm(p1 - center)
        a1 = math.degrees(math.atan2(p1[1]-cy, p1[0]-cx))
        a2 = math.degrees(math.atan2(p2[1]-cy, p2[0]-cx))
        
        # --- 短い方の弧(劣弧)を選択するロジック ---
        da = a2 - a1
        while da <= -180: da += 360
        while da > 180: da -= 360
        
        if da >= 0:
            theta1, theta2 = a1, a1 + da
        else:
            theta1, theta2 = a2, a2 + abs(da)
            
        if theta2 <= theta1: theta2 += 360
        # ---------------------------------------
        
        seg.update({'center': (cx, cy), 'radius': r, 'theta1': theta1, 'theta2': theta2})
        seg['artist'].center = (cx, cy)
        seg['artist'].width = seg['artist'].height = 2 * r
        seg['artist'].theta1, seg['artist'].theta2 = theta1, theta2
        self.mark_dirty()
        self.canvas.draw_idle()

    def set_axes_limits(self, xmin, xmax, ymin, ymax):
        self.axes.set_xlim(xmin, xmax)
        self.axes.set_ylim(ymin, ymax)
        self.canvas.draw_idle()

    def get_loop_orientation(self):
        """ループの回転方向を判定する (Shoelace公式による面積法)"""
        if not self.loop_closed or not self.segments: return None
        
        # サイン付き面積の計算 (外積の和)
        area = 0.0
        for seg in self.segments:
            p1 = self.points[seg['points'][0]]
            p2 = self.points[seg['points'][1]]
            # 外積 (x1*y2 - x2*y1) の累積
            area += (p1[0] * p2[1] - p2[0] * p1[1])
            
        if area > 1e-12: return "CCW"
        if area < -1e-12: return "CW"
        return "Unknown"

    def get_point_physical_names(self):
        """点ごとの物理名を判定する。線の端点にその線の名前を付与し、Dirichletを最優先する。"""
        pt_phys = {}
        for seg in self.segments:
            name = seg.get('physical_name')
            if not name: continue
            for p_idx in seg['points']:
                current = pt_phys.get(p_idx)
                if current == "Dirichlet": continue # 既にDirichletなら上書きしない
                pt_phys[p_idx] = name
        return pt_phys

    def export_geo(self, filepath, mesh_size, unit, mesh_order=1):
        scale = {"m": 1.0, "cm": 0.01, "mm": 0.001, "inch": 0.0254}.get(unit, 1.0)
        msh_filepath = os.path.splitext(filepath)[0] + ".msh"
        try:
            with open(filepath, 'w') as f:
                f.write(f"// Generated by Gmsh UI\n// Unit: {unit}\nlc = {mesh_size * scale};\n")
                for i, p in enumerate(self.points):
                    f.write(f"Point({i+1}) = {{{p[0]*scale}, {p[1]*scale}, 0, lc}};\n")
                arc_c_tags, c_idx = {}, len(self.points) + 1
                for i, seg in enumerate(self.segments):
                    if seg['type'] == 'arc':
                        c = seg['center']
                        f.write(f"Point({c_idx}) = {{{c[0]*scale}, {c[1]*scale}, 0, lc}};\n")
                        arc_c_tags[i] = c_idx; c_idx += 1
                for i, seg in enumerate(self.segments):
                    p1, p2 = seg['points'][0]+1, seg['points'][1]+1
                    if seg['type'] == 'line': f.write(f"Line({i+1}) = {{{p1}, {p2}}};\n")
                    else: f.write(f"Circle({i+1}) = {{{p1}, {arc_c_tags[i]}, {p2}}};\n")
                if self.loop_closed and len(self.segments) >= 3:
                    l_tags = ",".join([str(i+1) for i in range(len(self.segments))])
                    f.write(f"Curve Loop(1) = {{{l_tags}}};\nPlane Surface(1) = {{1}};\n")
                
                # Physical Groups
                phys = {}
                if self.loop_closed and len(self.segments) >= 3:
                    phys["Domain"] = {"dim": 2, "tags": [1]}
                
                # Lines
                for i, seg in enumerate(self.segments):
                    name = seg.get('physical_name')
                    if name:
                        if name not in phys: phys[name] = {"dim": 1, "tags": []}
                        phys[name]["tags"].append(i+1)
                
                # Points
                pt_phys_names = self.get_point_physical_names()
                pt_groups = {} # name -> [tags]
                for p_idx, name in pt_phys_names.items():
                    if name not in pt_groups: pt_groups[name] = []
                    pt_groups[name].append(p_idx + 1)
                
                for name, d in phys.items():
                    dim_s, tag_s = ("Curve" if d["dim"] == 1 else "Surface"), ",".join(map(str, d["tags"]))
                    f.write(f'Physical {dim_s}("{name}") = {{{tag_s}}};\n')
                
                for name, tags in pt_groups.items():
                    tag_s = ",".join(map(str, tags))
                    f.write(f'Physical Point("{name}") = {{{tag_s}}};\n')

                if mesh_order == 2: f.write("\nMesh.ElementOrder = 2;\n")
                f.write("\nMesh 2;\n")
                msh_path_esc = msh_filepath.replace("\\", "\\\\")
                f.write(f'Save "{msh_path_esc}";\n')
            wx.MessageBox(f"Exported to {filepath}", "Success")
            dlg = wx.MessageDialog(self, f"Run Gmsh to generate mesh?\n({filepath})", "Run Gmsh?", wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_YES:
                try:
                    subprocess.run(f'gmsh "{filepath}"', shell=True, check=True)
                except Exception as e:
                    wx.MessageBox(f"Gmsh execution failed: {e}", "Error")
            dlg.Destroy()
        except Exception as e:
            wx.MessageBox(f"Export failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def export_msh_direct(self, filepath, mesh_size, unit, mesh_order=1):
        scale = {"m": 1.0, "cm": 0.01, "mm": 0.001, "inch": 0.0254}.get(unit, 1.0)
        lc = mesh_size * scale
        try:
            gmsh.initialize()
            gmsh.model.add(os.path.splitext(os.path.basename(filepath))[0])
            p_tags = [gmsh.model.geo.addPoint(p[0]*scale, p[1]*scale, 0, lc) for p in self.points]
            arc_c_tags = {i: gmsh.model.geo.addPoint(seg['center'][0]*scale, seg['center'][1]*scale, 0, lc) for i, seg in enumerate(self.segments) if seg['type'] == 'arc'}
            s_tags = []
            for i, seg in enumerate(self.segments):
                p1, p2 = p_tags[seg['points'][0]], p_tags[seg['points'][1]]
                if seg['type'] == 'line': s_tags.append(gmsh.model.geo.addLine(p1, p2))
                else: s_tags.append(gmsh.model.geo.addCircleArc(p1, arc_c_tags[i], p2))
            surf_tag = -1
            if self.loop_closed and len(s_tags) >= 3:
                cl = gmsh.model.geo.addCurveLoop(s_tags)
                surf_tag = gmsh.model.geo.addPlaneSurface([cl])
            phys = {}
            if surf_tag > 0: phys["Domain"] = (2, [surf_tag])
            for i, seg in enumerate(self.segments):
                name = seg.get('physical_name')
                if name:
                    if name not in phys: phys[name] = (1, [])
                    phys[name][1].append(s_tags[i])
            for name, (dim, tags) in phys.items():
                g = gmsh.model.addPhysicalGroup(dim, tags)
                gmsh.model.setPhysicalName(dim, g, name)
            
            # Points Physical Groups
            pt_phys_names = self.get_point_physical_names()
            pt_groups = {} # name -> [tags]
            for p_idx, name in pt_phys_names.items():
                if name not in pt_groups: pt_groups[name] = []
                pt_groups[name].append(p_tags[p_idx])
            
            for name, tags in pt_groups.items():
                g = gmsh.model.addPhysicalGroup(0, tags)
                gmsh.model.setPhysicalName(0, g, name)

            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2 if surf_tag > 0 else 1)
            if mesh_order == 2: gmsh.model.mesh.setOrder(2)
            gmsh.write(filepath)
            wx.MessageBox(f"Mesh generated: {filepath}", "Success")
        except Exception as e:
            wx.MessageBox(f"Gmsh SDK error: {e}", "Error")
        finally:
            if gmsh.isInitialized(): gmsh.finalize()

    def export_python_script(self, filepath, mesh_size, unit, mesh_order=1):
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        code = [
            "import gmsh", "import math", "import os", "", "gmsh.initialize()", f'gmsh.model.add("{model_name}")', "",
            f"mesh_size = {mesh_size}", f'unit = "{unit}"',
            "scale = {'m': 1.0, 'cm': 0.01, 'mm': 0.001, 'inch': 0.0254}.get(unit, 1.0)",
            "lc = mesh_size * scale", "", "try:", "    p_tags = []"
        ]
        for i, p in enumerate(self.points):
            code.append(f"    p_tags.append(gmsh.model.geo.addPoint({p[0]}*scale, {p[1]}*scale, 0, lc))")
        
        code.append("    arc_c_tags = {}")
        for i, seg in enumerate(self.segments):
            if seg['type'] == 'arc':
                c = seg['center']
                code.append(f"    arc_c_tags[{i}] = gmsh.model.geo.addPoint({c[0]}*scale, {c[1]}*scale, 0, lc)")
        
        code.append("    s_tags = []")
        for i, seg in enumerate(self.segments):
            p1, p2 = seg['points'][0], seg['points'][1]
            if seg['type'] == 'line':
                code.append(f"    s_tags.append(gmsh.model.geo.addLine(p_tags[{p1}], p_tags[{p2}]))")
            else:
                code.append(f"    s_tags.append(gmsh.model.geo.addCircleArc(p_tags[{p1}], arc_c_tags[{i}], p_tags[{p2}]))")
        
        # Physical Groups aggregation logic
        code.append("    # Define Physical Groups")
        phys_groups = {}
        if self.loop_closed and len(self.segments) >= 3:
            code.append("    cl = gmsh.model.geo.addCurveLoop(s_tags)")
            code.append("    surf = gmsh.model.geo.addPlaneSurface([cl])")
            phys_groups["Domain"] = (2, ["surf"])
        
        for i, seg in enumerate(self.segments):
            name = seg.get('physical_name')
            if name:
                if name not in phys_groups: phys_groups[name] = (1, [])
                phys_groups[name][1].append(f"s_tags[{i}]")
        
        for name, (dim, tags) in phys_groups.items():
            tags_str = ", ".join(tags)
            code.append(f"    g_{name.replace(' ', '_')} = gmsh.model.addPhysicalGroup({dim}, [{tags_str}])")
            code.append(f"    gmsh.model.setPhysicalName({dim}, g_{name.replace(' ', '_')}, '{name}')")

        # Points Physical Groups
        code.append("    # Define Physical Groups for Points")
        pt_phys_names = self.get_point_physical_names()
        pt_groups = {} # name -> [tags]
        for p_idx, name in pt_phys_names.items():
            if name not in pt_groups: pt_groups[name] = []
            pt_groups[name].append(f"p_tags[{p_idx}]")
        
        for name, tags in pt_groups.items():
            tags_str = ", ".join(tags)
            g_var_name = f"g_pt_{name.replace(' ', '_')}"
            code.append(f"    {g_var_name} = gmsh.model.addPhysicalGroup(0, [{tags_str}])")
            code.append(f"    gmsh.model.setPhysicalName(0, {g_var_name}, '{name}')")

        code.append("    gmsh.model.geo.synchronize()")
        code.append(f"    gmsh.model.mesh.generate({2 if self.loop_closed else 1})")
        if mesh_order == 2:
            code.append("    gmsh.model.mesh.setOrder(2)")
        code.extend([f"    gmsh.write('{model_name}.msh')", "except Exception as e: print(f'Error: {e}')", "finally: gmsh.finalize()"])
        
        with open(filepath, 'w') as f:
            f.write("\n".join(code))
        wx.MessageBox(f"Python script exported: {filepath}", "Success")

    def import_superfish(self, filepath, unit):
        """Superfish (.af) ファイルを読み込んでジオメトリを再構築する"""
        scale = {"m": 0.01, "cm": 1.0, "mm": 10.0, "inch": 1.0/2.54}.get(unit, 1.0)
        line_re = re.compile(r"^\$po\s+x=([+\-\d.eE]+)\s*,\s*y=([+\-\d.eE]+)\s*\$", re.IGNORECASE)
        arc_re = re.compile(r"^\$po\s+nt=2\s*,\s*r=([+\-\d.eE]+)\s*,\s*theta=([+\-\d.eE]+)\s*,\s*x0=([+\-\d.eE]+)\s*,\s*y0=([+\-\d.eE]+)\s*\$", re.IGNORECASE)
        self.reset()
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line.lower().startswith("$po"): continue
                    lm, am = line_re.match(line), arc_re.match(line)
                    if not self.points and lm:
                        px, py = float(lm.group(1))*scale, float(lm.group(2))*scale
                        self.points.append((px, py))
                        self.markers.append(self.axes.plot(px, py, 'ro', markersize=5, picker=5)[0])
                        continue
                    if not self.points: continue
                    p1_idx = len(self.points) - 1
                    p1 = self.points[p1_idx]
                    if am:
                        # --- 円弧のインポートロジック (最短経路選択) ---
                        r_val = float(am.group(1)) * scale
                        theta_deg_end = float(am.group(2))
                        cx_val = float(am.group(3)) * scale
                        cy_val = float(am.group(4)) * scale
                        
                        # 終点の座標を計算
                        tr_end = math.radians(theta_deg_end)
                        px, py = cx_val + r_val * math.cos(tr_end), cy_val + r_val * math.sin(tr_end)
                        
                        # 始点の角度を取得
                        a1 = math.degrees(math.atan2(p1[1] - cy_val, p1[0] - cx_val))
                        a2 = theta_deg_end
                        
                        # 最短経路（180度以内）を計算するための正規化
                        da = a2 - a1
                        while da <= -180: da += 360
                        while da > 180: da -= 360
                        
                        if da >= 0:
                            theta1, theta2 = a1, a1 + da
                        else:
                            theta1, theta2 = a2, a2 + abs(da)
                        
                        if theta2 <= theta1: theta2 += 360
                        # ---------------------------------------------

                        self.points.append((px, py))
                        p2_idx = len(self.points) - 1
                        self.markers.append(self.axes.plot(px, py, 'ro', markersize=5, picker=5)[0])
                        
                        arc = patches.Arc((cx_val, cy_val), width=2*r_val, height=2*r_val, angle=0, 
                                          theta1=theta1, theta2=theta2, color='green', linewidth=1.0)
                        self.axes.add_patch(arc)
                        self.segments.append({'type': 'arc', 'points': [p1_idx, p2_idx], 'center': (cx_val, cy_val),
                                              'radius': r_val, 'theta1': theta1, 'theta2': theta2, 'artist': arc, 'physical_name': 'PEC'})
                    elif lm:
                        px, py = float(lm.group(1))*scale, float(lm.group(2))*scale
                        self.points.append((px, py))
                        p2_idx = len(self.points) - 1
                        self.markers.append(self.axes.plot(px, py, 'ro', markersize=5, picker=5)[0])
                        la = self.axes.plot([p1[0], px], [p1[1], py], 'b-')[0]
                        self.segments.append({'type': 'line', 'points': [p1_idx, p2_idx], 'artist': la, 'physical_name': 'PEC'})
            
            if len(self.points) > 2 and np.allclose(self.points[0], self.points[-1], atol=1e-7*scale):
                self.segments[-1]['points'][1] = 0
                self.points.pop(); self.markers.pop().remove()
                self.loop_closed = True
            self.canvas.draw(); self.mark_dirty()
            wx.MessageBox(f"Successfully imported from {filepath}", "Success")
        except Exception as e:
            wx.MessageBox(f"Failed to import Superfish file: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def export_superfish(self, filepath, mesh_size, unit):
        if not self.loop_closed:
            wx.MessageBox("Loop must be closed for Superfish.", "Error")
            return
        scale = {"m": 100.0, "cm": 1.0, "mm": 0.1, "inch": 2.54}.get(unit, 1.0)
        pts_cm = np.array(self.points) * scale; xmin, ymin = pts_cm.min(axis=0); xmax, ymax = pts_cm.max(axis=0)
        xdri, ydri = (xmin + xmax)/2.0, (ymin + ymax)/2.0
        try:
            with open(filepath, 'w') as f:
                f.write("Gmsh2Fish\n"); f.write(f"$reg kprob=1, dx={mesh_size*scale:.6e}, xdri={xdri:.6e}, ydri={ydri:.6e}, nbsup=1, nbslo=0, nbslf=0, nbsrt=0, freq=2856.0, kmethod=1, beta=1.0 $\n\n")
                p0 = self.points[0]; f.write(f"$po x={p0[0]*scale:.6e}, y={p0[1]*scale:.6e} $\n")
                for seg in self.segments:
                    p2 = self.points[seg['points'][1]]
                    if seg['type'] == 'line': f.write(f"$po x={p2[0]*scale:.6e}, y={p2[1]*scale:.6e} $\n")
                    else:
                        c, r = seg['center'], seg['radius']; th = math.degrees(math.atan2(p2[1]-c[1], p2[0]-c[0]))
                        f.write(f"$po nt=2, r={r*scale:.6e}, theta={th:.6e}, x0={c[0]*scale:.6e}, y0={c[1]*scale:.6e} $\n")
            wx.MessageBox(f"Superfish file exported: {filepath}", "Success")
        except Exception as e:
            wx.MessageBox(f"Export failed: {e}", "Error")
