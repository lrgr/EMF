#!/usr/bin/env python

################################################################################
# SETUP
################################################################################
import sys, os, logging

################################################################################
# LOGGING
################################################################################
#!/usr/bin/env

import sys, logging

def get_logger(verbosity=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(verbosity)
    return logger

def setup_logging(fp=None, level=logging.INFO):
    '''
    Setup logging with filepath
    '''
    FORMAT = '%(asctime)s %(levelname)-10s: %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if fp is not None:
        handlers.append(logging.FileHandler(fp))
    logging.basicConfig(format=FORMAT, level=level, handlers=handlers)

def log_dict(log_fn, d, prefix='', ntabs=1, dps=3):
    float_fmt = '{{:.{dps}f}}'.format(dps=dps)
    
    tabs = '\t' * ntabs
    fmt_string = '{tabs}{prefix} {{{{}}}}: {{fmt}}'.format(tabs=tabs, prefix=prefix)
    ffmt_string = fmt_string.format(fmt=float_fmt)
    default_fmt = fmt_string.format(fmt='{}')

    for k, v in sorted(d.items()):
        if isinstance(v, float):
            item_fmt = ffmt_string
        else:
            item_fmt = default_fmt
        log_fn(item_fmt.format(k,v))
        