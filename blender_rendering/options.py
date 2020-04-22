import argparse


class Options:
    def __init__(self, argv):

        if "--" not in argv:
            self.argv = []  # as if no args are passed
        else:
            self.argv = argv[argv.index("--") + 1:]

        usage_text = (
                "Run blender in background mode with this script:"
                "  blender --background --python [main_python_file] -- [options]"
        )

        self.parser = argparse.ArgumentParser(description=usage_text)
        self.initialize()
        self.args = self.parser.parse_args(self.argv)

    def initialize(self):
        self.parser.add_argument('--bvh_path', type=str, default='./example.bvh', help='path of input bvh file')
        self.parser.add_argument('--save_path', type=str, default='./results/', help='path of output video file')
        self.parser.add_argument('--render_engine', type=str, default='cycles',
                                 help='name of preferable render engine: cycles, eevee')
        self.parser.add_argument('--render', action='store_true', default=False, help='render an output video')
        # rendering parameters
        self.parser.add_argument('--frame_end', type=int, default=100, help='the index of the last rendered frame')
        self.parser.add_argument('--resX', type=int, default=960, help='x resolution')
        self.parser.add_argument('--resY', type=int, default=540, help='y resolution')


    def parse(self):
            return self.args
