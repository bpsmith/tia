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


class DeferredExecutionMixin(object):
    """Mixin which defers execution of methods by adding then to a queue until 'apply' is invoked or the object is
    invoked '()'.
    Don't want to use this if you modify object variables between method calls (Deferred calls methods later)
    """
    NOT_DEFERRED = ['apply']

    def __init__(self):
        self._deferred = []

    def __getattribute__(self, name):
        attr = super(DeferredExecutionMixin, self).__getattribute__(name)
        if callable(attr) and not name.startswith('_') and name not in self.NOT_DEFERRED \
                and not isinstance(attr, DeferredExecutionMixin):
            def wrapped(*args, **kwargs):
                self._deferred.append(lambda: attr(*args, **kwargs))
                return self

            return wrapped
        else:
            return attr

    def __call__(self):
        [f() for f in self._deferred]