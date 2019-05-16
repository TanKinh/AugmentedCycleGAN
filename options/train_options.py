from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--name', type=str, required=True, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/',help='models are saved here')

        # data
        self.parser.add_argument('--numpy_data', type=int, choices=[0,1], default=1, help='use numpy data instead of default JPEG images')

        # exp
        self.parser.add_argument('--seed', type=int, help='manual seed')
        self.parser.add_argument('--model', type=str, choices=['cycle_gan', 'stoch_cycle_gan', 'aug_cycle_gan'], default='aug_cycle_gan', help='which model to train')

        # supervised training
        self.parser.add_argument('--supervised', action='store_true', help='fraction of training data for supervised training')
        self.parser.add_argument('--sup_frac', type=float, default=0.1, help='fraction of training data for supervised training')
        self.parser.add_argument('--lambda_sup_A', type=float, default=0.1, help='weight for supervised loss (B -> A)')
        self.parser.add_argument('--lambda_sup_B', type=float, default=0.1, help='weight for supervised loss (A -> B)')

        # training
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--niter', type=int, default=25, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=25, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # model
        self.parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        self.parser.add_argument('--nef', type=int, default=32, help='# of encoder filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--nlatent', type=int, default=16, help='# of latent code dimensions. Used only for stochastic models, e.g. cycle_ali')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet', help='selects model to use for netG')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_gnorm', type=float, default=500., help='max grad norm to which it will be clipped (if exceeded)')
        self.parser.add_argument('--stoch_enc', action='store_true', help='use a stochastic encoder')
        self.parser.add_argument('--z_gan', type=int, default=1, choices=[0,1], help='use a GAN on z_B')
        self.parser.add_argument('--enc_A_B', type=int, default=1, choices=[0,1], help='encoder of z_B conditoned on both A and B')

        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_z_B', type=float, default=0.025, help='weight for cycle loss (B -> A -> B)')

        # monitoring
        self.parser.add_argument('--monitor_gnorm', type=bool, default=True, help='flag set to monitor grad norms')
        self.parser.add_argument('--display_freq', type=int, default=50000, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--num_multi', type=int, default=10, help='the number of z_B used to generate different B')
        self.parser.add_argument('--eval_A_freq', type=int, default=1, help='frequency of evaluating on A')
        self.parser.add_argument('--eval_B_freq', type=int, default=1, help='frequency of evaluating on B')

        self.initialized = True
