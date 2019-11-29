
import logging

log = logging.getLogger(__name__)

def task_plot():
    """ says hello """

    logging.basicConfig(level=logging.INFO)

    def action():
        log.info("Hallo Welt")

    return {
        "actions": [action]
    }