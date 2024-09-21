import setup
import argparse


def main():
    parser = argparse.ArgumentParser(description='CycleGAN Illumination Training and Testing Script')
    parser.add_argument('--data_root', type=str, default='dataset/', help='Root directory of the dataset')
    parser.add_argument('--save_model_path', type=str, default='models/', help='Directory to save trained models')
    parser.add_argument('--dataset_type', type=str, choices=['Portal', 'Blender'],default='Portal', help='Dataset type to be trained or tested')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training/trained epochs')
    parser.add_argument('--output_dir', type=str, default='results/', help='Directory to save generated images')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test', help='Operation mode: train or test')

    args = parser.parse_args()

    if args.mode == 'train':
        setup.experiment_CycleGAN(output_dir=args.output_dir,
                                  epochs=args.epochs,
                                  saved_location=None,
                                  trained_epoch=0,
                                  dataType=args.dataset_type
                                  )
    else:
        setup.experiment_CycleGAN(output_dir=args.output_dir,
                                  epochs=0,
                                  saved_location=args.save_model_path,
                                  trained_epoch=args.epochs,
                                  dataType=args.dataset_type
                                  )

if __name__=="__main__":
    main()