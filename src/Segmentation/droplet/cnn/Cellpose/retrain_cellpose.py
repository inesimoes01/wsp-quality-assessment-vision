from cellpose import models,core, io

from cellpose import io, models, train 
import torch 


if __name__ == "__main__":
    output = io.load_train_test_data(train_dir='/test/', 
                                 image_filter='_img', 
                                 mask_filter='_masks', 
                                 look_one_level_down=False)
    images, labels, image_names, test_images, test_labels, image_names_test = output
    device = torch.device('cuda')
    model = models.CellposeModel(model_type='cyto3', pretrained_model='cyto3', device = device)

    model_path = train.train_seg(
        model.net, 
        train_data=images, 
        train_labels=labels, 
        channels=[0,0],
        normalize=True, 
        weight_decay=1e-4,
        SGD=False, 
        learning_rate=0.1, 
        n_epochs=1000, 
        save_path='/workspace/fine-tune-cellpose',
        model_name='Cellpose_fine_tuned_1000_pretrain.pth',
        min_train_masks = 1
    )

    print('Done')