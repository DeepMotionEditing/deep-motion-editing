from options.options import Options
from retargeting.models import create_model as create_model_retargeting
from style_transfer.models import create_model as create_model_style_transfer
from models import create_model

if __name__ == '__main__':
    opt = Options().parse()

    motion_A = read_motion(opt.input_A)
    motion_B = read_motion(opt.input_B)

    model = create_model(opt.model_path, opt.edit_type)

    output = model.test(motion_A, motion_B)

    save_motion(opt.result_dir)


def create_model(model_path, edit_type):
    if edit_type == "retargeting":
        return create_model_retargeting(model_path)
    elif edit_type == "style_transfer":
        return create_model_style_transfer(model_path)
    else:
        raise ValueError("Unsupported editing type.")

def read_motion(path):
    if path.lower().endswith('.bvh'):
        bvh = read_bvh(path)
        return bvh_to_animation(bvh)

    elif path.lower().endswith('.json'):
        json = read_json(path)
        return json_to_motion(json)

    else:
        raise ValueError("Unsupported file extension.")
