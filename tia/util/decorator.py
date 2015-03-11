from functools import wraps

def lazy_property(fct, name=None):
    name = name or fct.__name__
    attr_name = '_' + name
    if attr_name == '_<lambda>':
        raise Exception("cannot assign <lambda> to lazy property")

    @property
    @wraps(fct)
    def _wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fct(self))
        return getattr(self, attr_name)
    return _wrapper
