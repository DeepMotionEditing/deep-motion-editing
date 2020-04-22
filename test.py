from options.options import Options
from retargeting.models import create_model as create_model_retargeting
from style_transfer.models import create_model as create_model_style_transfer
from deep_animation import animation_2D, animation_3D

if __name__ == '__main__':
    opt = Options().parse()

    animation_A = read_animation(opt.input_A)
    animation_B = read_animation(opt.input_B)

    model = create_model(opt.model_path, opt.edit_type)

    output = model.test(animation_A, animation_B)

    output_corrected = foot_contact_correct(output, animation_A.foot_contact)
    save_animation(output_corrected, opt.result_dir)


def create_model(model_path, edit_type):
    if edit_type == "retargeting":
        return create_model_retargeting(model_path)
    elif edit_type == "style_transfer":
        return create_model_style_transfer(model_path)
    else:
        raise ValueError("Unsupported editing type.")

def read_animation(path):
    if path.lower().endswith('.bvh'):
        bvh = read_bvh(path)
        return bvh_to_animation(bvh)

    elif path.lower().endswith('.json'):
        json = read_json(path)
        return json_to_2D_animation(json)

    else:
        raise ValueError("Unsupported animation file extension.")
