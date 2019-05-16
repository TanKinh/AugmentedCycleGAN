from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--chk_path', required=True, type=str, help='path to checkpoint -- we assume expr_dir is containing dir')
        self.parser.add_argument('--res_dir', type=str, default='test_res', help='results directory (will create under expr_dir)')
        self.parser.add_argument('--train_logvar', type=int, default=1, help='train logvar_B on training data')
        self.parser.add_argument('--metric', required=True, type=str, choices=['bpp', 'mse', 'visual', 'noise_sens'])