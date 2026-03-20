import logging
import os

def setup_logger(save_dir, name="vqgan"):
    """设置日志（控制台+文件）"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # 文件处理器
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, f"{name}.log"))
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
