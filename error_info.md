(ece572) zju_earthdata@ZJU-EarthData:~/code_repo/ECE572_Final_Project$ python visualization.py 
Files already downloaded and verified
Processing clean_model...
/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
/home/zju_earthdata/code_repo/ECE572_Final_Project/visualization.py:10: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path, map_location=device))
Sample 1: Label=3
Traceback (most recent call last):
  File "/home/zju_earthdata/code_repo/ECE572_Final_Project/visualization.py", line 137, in <module>
    visualize_cam(img, cam, title=f"{model_name.upper()} - CAM for Sample {i+1}", save_path=cam_save_path)
  File "/home/zju_earthdata/code_repo/ECE572_Final_Project/visualization.py", line 58, in visualize_cam
    plt.imshow(img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # 原始图像
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/pyplot.py", line 3562, in imshow
    __ret = gca().imshow(
            ^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/pyplot.py", line 2742, in gca
    return gcf().gca()
           ^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/pyplot.py", line 1093, in gcf
    return figure()
           ^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/pyplot.py", line 1027, in figure
    manager = new_figure_manager(
              ^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/pyplot.py", line 550, in new_figure_manager
    return _get_backend_mod().new_figure_manager(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/backend_bases.py", line 3507, in new_figure_manager
    return cls.new_figure_manager_given_figure(num, fig)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/backend_bases.py", line 3512, in new_figure_manager_given_figure
    return cls.FigureCanvas.new_manager(figure, num)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/backend_bases.py", line 1797, in new_manager
    return cls.manager_class.create_with_canvas(cls, figure, num)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/backends/_backend_tk.py", line 504, in create_with_canvas
    manager = cls(canvas, num, window)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/backends/_backend_tk.py", line 457, in __init__
    super().__init__(canvas, num)
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/backend_bases.py", line 2655, in __init__
    self.toolbar = self._toolbar2_class(self.canvas)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/backends/_backend_tk.py", line 624, in __init__
    self._buttons[text] = button = self._Button(
                                   ^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/backends/_backend_tk.py", line 825, in _Button
    NavigationToolbar2Tk._set_image_for_button(self, b)
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/matplotlib/backends/_backend_tk.py", line 761, in _set_image_for_button
    image = ImageTk.PhotoImage(im.resize((size, size)), master=self)
                               ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/PIL/Image.py", line 2342, in resize
    im = im.resize(size, resample, box)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/PIL/Image.py", line 2365, in resize
    return self._new(self.im.resize(size, resample, box))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: height and width must be > 0
