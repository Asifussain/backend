import argparse
import torch
import os # Import os for path operations if needed later
from exp.exp_classification import Exp_Classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADformer')

    # Basic config
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=0, help='status (0 for prediction)')
    parser.add_argument('--model_id', type=str, default='ADSZ-Indep', help='model id (used for setting string)')
    parser.add_argument('--model', type=str, default='ADformer', help='model name')

    # Data loader parameters (some might not be strictly needed for predict_unlabeled_sample but define structure)
    parser.add_argument('--data', type=str, default='ADSZIndep', help='dataset type (used for setting string)')
    parser.add_argument('--features', type=str, default='M',
                        help='features [M, S, MS] (used for setting string)')
    parser.add_argument('--target', type=str, default='OT', help='target feature (usually not needed for predict)')
    parser.add_argument('--freq', type=str, default='h', help='frequency for time encoding (used by some models)')
    # Removed the --checkpoints argument as the path is constructed in exp_classification.py using 'setting'

    # Input file for prediction (this is passed from app.py)
    parser.add_argument('--input_file', type=str, required=True, # Make it required if not providing default
                        help='Path to input .npy file')

    # Forecasting task parameters (might influence model loading if architecture depends on them)
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length (used for setting string & model)')
    parser.add_argument('--label_len', type=int, default=48, help='start token length (used for setting string)')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length (used for setting string)')
    # parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='seasonal patterns') # Likely not needed for predict
    # parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False) # Likely not needed for predict

    # Model parameters (passed to the model constructor)
    parser.add_argument('--enc_in', type=int, default=19, help='encoder input size (number of channels)')
    # parser.add_argument('--dec_in', type=int, default=7, help='decoder input size') # Usually not needed for classification predict
    parser.add_argument('--num_class', type=int, default=2, help='number of classes')
    # parser.add_argument('--c_out', type=int, default=7, help='output size') # Usually num_class for classification
    parser.add_argument('--d_model', type=int, default=128, help='model dimension (used for setting string)')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads (used for setting string)')
    parser.add_argument('--e_layers', type=int, default=6, help='number of encoder layers (used for setting string)')
    parser.add_argument('--d_layers', type=int, default=1, help='number of decoder layers (used for setting string)')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of FFN (used for setting string)')
    # parser.add_argument('--moving_avg', type=int, default=25, help='moving average window') # Model specific?
    parser.add_argument('--factor', type=int, default=1, help='attention factor (used for setting string)')
    parser.add_argument('--distil', action='store_false', help='disable distilling', default=True) # Used for setting string
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--embed', type=str, default='timeF', help='embedding type (used for setting string)')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--output_attention', action='store_true', help='output attention flag')
    # parser.add_argument('--chunk_size', type=int, default=16, help='chunk size') # Model specific?
    # parser.add_argument('--patch_len', type=int, default=16, help='patch length') # Model specific?
    # parser.add_argument('--stride', type=int, default=8, help='stride') # Model specific?
    # parser.add_argument('--sampling_rate', type=int, default=256, help='sampling rate') # Needed for visualization, potentially pass to Exp?

    # ADformer specific args (as passed from app.py)
    parser.add_argument("--patch_len_list", type=str, default='4', help="list of patch lengths")
    parser.add_argument("--up_dim_list", type=str, default='19', help="list of upsampling dimensions")
    parser.add_argument("--augmentations", type=str, default="none", help="augmentation types (likely not used in predict)")
    parser.add_argument("--no_inter_attn", action="store_true", help="disable inter-attention", default=False)
    parser.add_argument("--no_temporal_block", action="store_true", help="disable temporal block", default=False)
    parser.add_argument("--no_channel_block", action="store_true", help="disable channel block", default=False)

    # Optimization parameters (mostly for training, not prediction)
    # parser.add_argument('--num_workers', type=int, default=0, help='number of data loader workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments iterations (loop in main)')
    # parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (might be overridden for prediction)')
    # parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='experiment description (used for setting string)')
    parser.add_argument('--seed', type=int, default=None, help='optional seed used in checkpoint naming (appended to setting)')
    # parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # parser.add_argument('--lradj', type=str, default='type1', help='learning rate adjustment type')
    # parser.add_argument('--use_amp', action='store_true', help='use AMP for training', default=False)
    parser.add_argument("--swa", action="store_true", help="use Stochastic Weight Averaging", default=False) # Relevant for loading SWA model
    # parser.add_argument('--no_normalize', action='store_true', help='do not normalize data', default=False) # Normalization usually handled outside

    # GPU parameters (Leave this logic as is)
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu if available (overridden by device check)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple GPUs', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='comma-separated GPU device ids')

    args = parser.parse_args()

    # Decide device: use GPU if available and requested; else CPU.
    # Keep this logic unchanged as requested
    args.use_gpu = args.use_gpu and torch.cuda.is_available() # Force False if cuda not available
    device = torch.device("cuda" if args.use_gpu else "cpu")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0] # Use the first ID for single-GPU operations if needed
        # Note: DataParallel handles multi-GPU distribution in _build_model
    else:
         args.use_multi_gpu = False
         args.device_ids = [] # Ensure it's empty if not using multi-gpu

    print(f"Using Device: {device}")

    # --- Setting string construction - used in exp_classification for path ---
    # Ensure this format exactly matches the folder name containing checkpoint.pth
    # Based on console log: 'ADSZ-IndepftM_sl128_ll48_pl96_dm128_nh8_el6_dl1_df256_fc1_ebtimeF_dtTrueExp'
    # It seems to combine many args. The original code's format function does this.
    # Make sure args passed from app.py result in the correct string here.
    # In backend/SIDDHI/run.py
    setting = '{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.model_id,       # e.g., ADSZ-Indep
        args.features,       # e.g., M  <- MUST be passed correctly now
        args.seq_len,        # e.g., 96
        args.label_len,      # e.g., 48 <- MUST be passed correctly now
        args.pred_len,       # e.g., 96 <- MUST be passed correctly now
        args.d_model,        # e.g., 128
        args.n_heads,        # e.g., 8  <- MUST be passed correctly now
        args.e_layers,       # e.g., 6
        args.d_layers,       # e.g., 1  <- MUST be passed correctly now
        args.d_ff,           # e.g., 256
        args.factor,         # e.g., 1  <- MUST be passed correctly now
        args.embed,          # e.g., timeF <- MUST be passed correctly now
        args.distil,         # e.g., True <- MUST be passed correctly now (as string 'True' or bool True)
        args.des             # e.g., Exp <- MUST be passed correctly now
    )
    if args.seed is not None:
        setting = f"{setting}_seed{args.seed}"
    # --- End Setting String Construction ---


    Exp = Exp_Classification # Assign class, not instantiate yet

    # Initialize and run prediction
    # The loop `for ii in range(args.itr)` seems unnecessary for single prediction
    # We'll run it once based on the arguments passed.
    exp = Exp(args)  # Create experiment instance, passes all args
    print('>>>>>>> Testing on given sample using setting: {} <<<<<<<<'.format(setting))

    # Make sure the 'root_path' argument exists in args if exp_classification expects it for saving output.json
    # If not defined, set a default relative path (e.g., current dir '.')
    if not hasattr(args, 'root_path') or args.root_path is None:
         args.root_path = '.' # Default to current directory (SIDDHI)

    # Pass the determined device to the prediction function
    # The predict_unlabeled_sample method in exp_classification now uses 'setting' to find the checkpoint
    exp.predict_unlabeled_sample(args.input_file, setting, device)

    if args.use_gpu: # Only empty cache if GPU was used
        torch.cuda.empty_cache()