__author__ = 'Marek'


class AlphaMatte:
    ALPHA_BACKGROUND = 0
    ALPHA_FOREGROUND = 1

    TRIMAP_FOREGROUND = 1
    TRIMAP_BACKGROUND = -1
    TRIMAP_UNKNOWN = 0

    def get_alpha(self, *args, **kwargs):
        raise NotImplemented("Method should be overridden.")


class Matting:
    def foreground(self, alpha_mask, *args, **kwargs):
        raise NotImplemented("Method should be overridden.")

    def background(self, alpha_mask, *args, **kwargs):
        raise NotImplemented("Method should be overridden.")

    def replace_foreground(self, new_foreground, alpha_mask, *args, **kwargs):
        raise NotImplemented("Method should be overridden.")

    def replace_background(self, new_background, alpha_mask, *args, **kwargs):
        raise NotImplemented("Method should be overridden.")
