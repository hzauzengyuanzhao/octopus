import torch

def get_model(model, model_path):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Clean the weight dictionary (remove the 'model.' prefix)
    new_state_dict = {
        k[len('model.'):] if k.startswith('model.') else k: v
        for k, v in state_dict.items()
    }

    print("Epoch:", checkpoint['epoch'])
    print("Val Loss:", checkpoint['val_loss'])
    print("Val Pearson:", checkpoint['val_pearson'])
    print("val_insu_corr:", checkpoint['val_insu_corr'])
    print("Observed vs expected:", checkpoint['val_oe'])
    model.load_state_dict(new_state_dict, strict=True)
    return model


def get_mapping_model(model, model_path, teacher_path=None):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    new_state_dict = {
        k[len('model.'):] if k.startswith('model.') else k: v
        for k, v in state_dict.items()
    }

    print("Epoch:", checkpoint.get('epoch', 'N/A'))
    print("Val Loss:", checkpoint.get('val_loss', 'N/A'))
    print("Val Pearson:", checkpoint.get('val_pearson', 'N/A'))
    print("val_insu_corr:", checkpoint.get('val_insu_corr', 'N/A'))
    print("Observed vs expected:", checkpoint.get('val_oe', 'N/A'))

    # Load main model parameters (excluding teacher)
    main_state_dict = {k: v for k, v in new_state_dict.items() if not k.startswith("teacher_model.")}
    model.load_state_dict(main_state_dict, strict=False)

    # If teacher_path is provided, load the teacher model
    if teacher_path is not None:
        teacher_checkpoint = torch.load(teacher_path, map_location="cpu")
        teacher_state_dict = teacher_checkpoint.get('model_state_dict', teacher_checkpoint)

        teacher_state_dict = {
            k[len('model.'):] if k.startswith('model.') else k: v
            for k, v in teacher_state_dict.items()
        }

        if model.teacher_model is not None:
            model.teacher_model.load_state_dict(teacher_state_dict, strict=False)
            print(f"Loaded teacher model from {teacher_path}")
        else:
            print("Warning: model has no teacher_model, but teacher_path was provided")

    else:
        # If no teacher_path,teacher model is None
        model.teacher_model = None
        print("No teacher model loaded.")
    return model



