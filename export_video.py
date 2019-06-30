# Source: util_scripts.py https://github.com/sara-nl/progressive_growing_of_gans
#Generates training progress video with training statistics.
import os
import re
import bisect
import numpy as np
import scipy.ndimage
import config
import train
import glob
import pickle
from collections import OrderedDict
import moviepy.editor # pip install moviepy

def main():
    # Configure run_id
    run_id = '00000'
    generate_training_video(run_id)

def generate_training_video(run_id, duration_sec=20, time_warp=1.5, mp4=None, mp4_fps=60, mp4_codec='libx265', mp4_bitrate='16M'):
    src_result_subdir = locate_result_subdir(run_id)
    if mp4 is None:
        mp4 = os.path.basename(src_result_subdir) + '-train.mp4'

    # Parse log.
    times = []
    snaps = [] # [(png, kimg, lod), ...]
    with open(os.path.join(src_result_subdir, 'log.txt'), 'rt') as log:
        for line in log:
            k = re.search(r'kimg ([\d\.]+) ', line)
            l = re.search(r'lod ([\d\.]+) ', line)
            t = re.search(r'time (\d+d)? *(\d+h)? *(\d+m)? *(\d+s)? ', line)
            if k and l and t:
                k = float(k.group(1))
                l = float(l.group(1))
                t = [int(t.group(i)[:-1]) if t.group(i) else 0 for i in range(1, 5)]
                t = t[0] * 24*60*60 + t[1] * 60*60 + t[2] * 60 + t[3]
                png = os.path.join(src_result_subdir, 'fakes%06d.png' % int(np.floor(k)))
                if os.path.isfile(png):
                    times.append(t)
                    snaps.append((png, k, l))
    assert len(times)

    # Frame generation func for moviepy.
    png_cache = [None, None] # [png, img]
    def make_frame(t):
        wallclock = ((t / duration_sec) ** time_warp) * times[-1]
        png, kimg, lod = snaps[max(bisect.bisect(times, wallclock) - 1, 0)]
        if png_cache[0] == png:
            img = png_cache[1]
        else:
            img = scipy.misc.imread(png)
            # Check if the img is greyscale. If so, it needs to be converted to RGB, since MoviePy expects it in that format
            if img.ndim == 2:
                tmpimg = img
                img = np.zeros([tmpimg.shape[0], tmpimg.shape[1], 3], tmpimg.dtype)
                img[:,:,0] = tmpimg
                img[:,:,1] = tmpimg
                img[:,:,2] = tmpimg
            while img.shape[1] > 1920 or img.shape[0] > 1080:
                img = img.astype(np.float32).reshape(img.shape[0]//2, 2, img.shape[1]//2, 2, -1).mean(axis=(1,3))
            png_cache[:] = [png, img]
        # Computes effective resolution from LoD.
        res = 1024*(2**-float(lod))
        img = draw_text_label(img, 'lod %.2f' % lod, 16, img.shape[0]-4, alignx=0.0, aligny=1.0)
        img = draw_text_label(img, 'res %.2f' % res, img.shape[0]//3, img.shape[0]-4, alignx=0.5, aligny=1.0)
        img = draw_text_label(img, format_time(int(np.rint(wallclock))), img.shape[1]//1.5, img.shape[0]-4, alignx=0.5, aligny=1.0)
        img = draw_text_label(img, '%.0f kimg' % kimg, img.shape[1]-16, img.shape[0]-4, alignx=1.0, aligny=1.0)
        return img

    # Generate video.
    result_subdir = create_result_subdir(config.result_dir, train.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()
    
def generate_interpolation_video(run_id, snapshot=None, grid_size=[1,1], image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, mp4=None, mp4_fps=60, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    network_pkl = locate_network_pkl(run_id, snapshot)
    if mp4 is None:
        mp4 = get_id_string_for_network_pkl(network_pkl) + '-lerp.mp4'
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = load_network_pkl(run_id, snapshot)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

_text_label_cache = OrderedDict()

def draw_text_label(img, text, x, y, alignx=0.5, aligny=0.5, color=255, opacity=1.0, glow_opacity=1.0, **kwargs):
    color = np.array(color).flatten().astype(np.float32)
    assert img.ndim == 3 and img.shape[2] == color.size or color.size == 1
    alpha, glow = setup_text_label(text, **kwargs)
    xx, yy = int(np.rint(x - alpha.shape[1] * alignx)), int(np.rint(y - alpha.shape[0] * aligny))
    xb, yb = max(-xx, 0), max(-yy, 0)
    xe, ye = min(alpha.shape[1], img.shape[1] - xx), min(alpha.shape[0], img.shape[0] - yy)
    img = np.array(img)
    slice = img[yy+yb : yy+ye, xx+xb : xx+xe, :]
    slice[:] = slice * (1.0 - (1.0 - (1.0 - alpha[yb:ye, xb:xe]) * (1.0 - glow[yb:ye, xb:xe] * glow_opacity)) * opacity)[:, :, np.newaxis]
    slice[:] = slice + alpha[yb:ye, xb:xe, np.newaxis] * (color * opacity)[np.newaxis, np.newaxis, :]
    return img

def setup_text_label(text, font='Calibri', fontsize=32, padding=6, glow_size=2.0, glow_coef=3.0, glow_exp=2.0, cache_size=100): # => (alpha, glow)
    # Lookup from cache.
    key = (text, font, fontsize, padding, glow_size, glow_coef, glow_exp)
    if key in _text_label_cache:
        value = _text_label_cache[key]
        del _text_label_cache[key] # LRU policy
        _text_label_cache[key] = value
        return value

    # Limit cache size.
    while len(_text_label_cache) >= cache_size:
        _text_label_cache.popitem(last=False)

    # Render text.
    import moviepy.editor # pip install moviepy
    alpha = moviepy.editor.TextClip(text, font=font, fontsize=fontsize).mask.make_frame(0)
    alpha = np.pad(alpha, padding, mode='constant', constant_values=0.0)
    glow = scipy.ndimage.gaussian_filter(alpha, glow_size)
    glow = 1.0 - np.maximum(1.0 - glow * glow_coef, 0.0) ** glow_exp

    # Add to cache.
    value = (alpha, glow)
    _text_label_cache[key] = value
    return value

def locate_result_subdir(run_id_or_result_subdir):
    if isinstance(run_id_or_result_subdir, str) and os.path.isdir(run_id_or_result_subdir):
        return run_id_or_result_subdir

    searchdirs = []
    searchdirs += ['']
    searchdirs += ['results']
    searchdirs += ['networks']

    for searchdir in searchdirs:
        dir = config.result_dir if searchdir == '' else os.path.join(config.result_dir, searchdir)
        dir = os.path.join(dir, str(run_id_or_result_subdir))
        if os.path.isdir(dir):
            return dir
        prefix = '%03d' % run_id_or_result_subdir if isinstance(run_id_or_result_subdir, int) else str(run_id_or_result_subdir)
        dirs = sorted(glob.glob(os.path.join(config.result_dir, searchdir, prefix + '-*')))
        dirs = [dir for dir in dirs if os.path.isdir(dir)]
        if len(dirs) == 1:
            return dirs[0]
    raise IOError('Cannot locate result subdir for run', run_id_or_result_subdir)

def create_result_subdir(result_dir, desc):

    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[:fbase.find('-')])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d-%s' % (run_id, desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print("Saving results to", result_subdir)
    set_output_log_file(os.path.join(result_subdir, 'log.txt'))

    # Export config.
    try:
        with open(os.path.join(result_subdir, 'config.txt'), 'wt') as fout:
            for k, v in sorted(config.__dict__.items()):
                if not k.startswith('_'):
                    fout.write("%s = %s\n" % (k, str(v)))
    except:
        pass

    return result_subdir

def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60:         return '%ds'                % (s)
    elif s < 60*60:    return '%dm %02ds'          % (s // 60, s % 60)
    elif s < 24*60*60: return '%dh %02dm %02ds'    % (s // (60*60), (s // 60) % 60, s % 60)
    else:         return '%dd %02dh %02dm'    % (s // (24*60*60), (s // (60*60)) % 24, (s // 60) % 60)

output_logger = None

def set_output_log_file(filename, mode='wt'):
    if output_logger is not None:
        output_logger.set_log_file(filename, mode)
        
def load_network_pkl(run_id_or_result_subdir_or_network_pkl, snapshot=None):
    return load_pkl(locate_network_pkl(run_id_or_result_subdir_or_network_pkl, snapshot))

def locate_network_pkl(run_id_or_result_subdir_or_network_pkl, snapshot=None):
    if isinstance(run_id_or_result_subdir_or_network_pkl, str) and os.path.isfile(run_id_or_result_subdir_or_network_pkl):
        return run_id_or_result_subdir_or_network_pkl

    pkls = list_network_pkls(run_id_or_result_subdir_or_network_pkl)
    if len(pkls) >= 1 and snapshot is None:
        return pkls[-1]
    for pkl in pkls:
        try:
            name = os.path.splitext(os.path.basename(pkl))[0]
            number = int(name.split('-')[-1])
            if number == snapshot:
                return pkl
        except ValueError: pass
        except IndexError: pass
    raise IOError('Cannot locate network pkl for snapshot', snapshot)
    
def list_network_pkls(run_id_or_result_subdir, include_final=True):
    result_subdir = locate_result_subdir(run_id_or_result_subdir)
    pkls = sorted(glob.glob(os.path.join(result_subdir, 'network-*.pkl')))
    if len(pkls) >= 1 and os.path.basename(pkls[0]) == 'network-final.pkl':
        if include_final:
            pkls.append(pkls[0])
        del pkls[0]
    return pkls
        
def get_id_string_for_network_pkl(network_pkl):
    p = network_pkl.replace('.pkl', '').replace('\\', '/').split('/')
    return '-'.join(p[max(len(p) - 2, 0):])
    
def load_pkl(filename):
    with open(filename, 'rb') as file:
        dnnlib.tflib.init_tf()
        return LegacyUnpickler(file, encoding='latin1').load()
        
class LegacyUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == 'network' and name == 'Network':
            return tfutil.Network
        return super().find_class(module, name)

    
if __name__ == "__main__":
    main()
