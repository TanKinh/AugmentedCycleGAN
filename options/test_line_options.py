from .base_options import BaseOptions

class TestLineOption(BaseOptions):
    def __init__(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--chk_path', required=True, type=str, help='path to checkpoint -- we assume expr_dir is containing dir')
        self.parser.add_argument('--res_dir', type=str, default='test_res', help='results directory (will create under expr_dir)')
        self.parser.add_argument('--metric', required=True, type=str, choices=['bpp', 'mse', 'visual', 'noise_sens'])
        self.parser.add_argument('--font_size', type=int, default=128, help="the font's character size")
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--offset', type=int, default=0, help="the font's character x and y offset")