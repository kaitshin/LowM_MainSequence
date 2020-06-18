import logging

formatter = logging.Formatter('%(asctime)s - %(module)12s.%(funcName)20s - %(levelname)s: %(message)s')


class MLog:
    """
    Main class to log information to stdout and ASCII file

    To execute:
    mylog = MLog(dir0)._get_logger()

    Parameters
    ----------
    dir0 : str
      Full path for where log files should be placed

    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 2 October 2019
    """

    def __init__(self, dir0, str_date):
        self.LOG_FILENAME = dir0 + 'completeness_analysis.' + str_date + '.log'
        self._log = self._get_logger()

    def _get_logger(self):
        loglevel = logging.INFO
        log = logging.getLogger(self.LOG_FILENAME)
        if not getattr(log, 'handler_set', None):
            log.setLevel(logging.INFO)
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            log.addHandler(sh)

            fh = logging.FileHandler(self.LOG_FILENAME)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)

            log.setLevel(loglevel)
            log.handler_set = True
        return log
