import logging
import logging.config


def default_setup():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,  # fixes issue with config set up after loading loggers
        'formatters': {
            'standard': {'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'},
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': True
            }
        }
    })


def get_logger(name, level=None):
    logger = logging.getLogger(name)
    level is not None and logger.setLevel(level)
    return logger


def class_logger(cls, level=None):
    lname = cls.__module__ + "." + cls.__name__
    return get_logger(lname, level)


def instance_logger(name, instance, level=None):
    lname = "%s.%s.%s" % (instance.__class__.__module__, instance.__class__.__name__, name)
    return get_logger(lname, level)

