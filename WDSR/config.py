import argparse

def get_args():
    parser = argparse.ArgumentParser(description='****Model Argparse.****')

    # Data and log dir
    parser.add_argument('--train_image', type=str, default='./data/train_iamge')
    parser.add_argument('--train_label', type=str, default='./data/train_label')
    parser.add_argument('--train_data', type=str, default='wdsr_train_data.h5')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--save_dir', type=str, default='./saves')

    # Train
    parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="params in Adam")
    parser.add_argument("--epoches", type=int, default=300, help="epoches for train")
    parser.add_argument("--test_directory_path", type=str, default='lr/', help="lr img for eval")
    parser.add_argument("--label_directory_path", type=str, default='hr/', help="hr img for eval")
    parser.add_argument("--psnr_max", type=int, default=0, help="max psnr init")
    parser.add_argument("--file", type=str, default='wdsr_train_data.h5', help="dataset file")
    parser.add_argument("--savepath", type=str, default='./checkpoint/flip_image/wdsrrgb_flip_image.pkl',
                        help="save path")

    #Test
    parser.add_argument("--test_type", type=str, default='B', help=
    "A: 超分辨率 并且把图片都保存 txt也有"
    "B: 超分辨率 只看总PSNR 指标")

    # parser.add_argument("--model_path", type=str, default='./checkpoint/flip_image/wdsr_checkpoint_240.pkl', help="wdsr model")
    parser.add_argument("--model_path", type=str, default='wdsrrgb.pkl', help="wdsr model")
    #parser.add_argument("--test_directory_path", type=str, default='lr/', help="lr img for eval")
    #parser.add_argument("--label_directory_path", type=str, default='hr/', help="hr img for eval")
    parser.add_argument("--txt_path", type=str, default='./result/index.txt', help="save psnr in txt")
    parser.add_argument("--save_path", type=str, default='./result/image/', help="save super resolution image")

    # test_directory_path = '/root/workspace/liuzhen/dataset/youku/data/test_l/'
    # label_directory_path = '/root/workspace/liuzhen/dataset/youku/data/test_h/'
    # txt_path = './result/flip_image_epoch_240/index.txt'
    # save_path = './result/flip_image_epoch_240/image/'

    return parser
