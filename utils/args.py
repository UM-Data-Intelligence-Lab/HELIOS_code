import six
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def print_arguments(args):
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in six.iteritems(vars(args)):
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')
