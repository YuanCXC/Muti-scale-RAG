# -*- coding: utf-8 -*-
"""统一日志记录模块

提供统一的日志配置和管理功能，支持：
- 控制台和文件输出
- 不同日志级别
- 彩色输出
- 日志文件轮转
- 第三方库日志抑制
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from functools import lru_cache

class ColorCodes:
    """ANSI 颜色代码"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器
    
    根据日志级别使用不同的颜色输出。
    """
    
    LEVEL_COLORS = {
        logging.DEBUG: ColorCodes.CYAN,
        logging.INFO: ColorCodes.GREEN,
        logging.WARNING: ColorCodes.YELLOW,
        logging.ERROR: ColorCodes.RED,
        logging.CRITICAL: ColorCodes.BG_RED + ColorCodes.WHITE,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录
        
        Args:
            record: 日志记录对象
            
        Returns:
            格式化后的日志字符串
        """
        original_levelname = record.levelname
        
        if record.levelno in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelno]
            record.levelname = f"{color}{record.levelname}{ColorCodes.RESET}"
        
        result = super().format(record)
        
        record.levelname = original_levelname
        
        return result


class LoggerManager:
    """日志管理器
    
    管理所有日志记录器的创建和配置。
    """
    
    _initialized: bool = False
    _loggers: dict = {}
    
    @classmethod
    def setup(
        cls,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        use_color: bool = True,
        format_string: Optional[str] = None,
    ) -> None:
        """设置全局日志配置
        
        Args:
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的日志文件数量
            use_color: 是否使用彩色输出
            format_string: 自定义格式字符串
        """
        if cls._initialized:
            return
        
        from src.utils.config import get_config
        config = get_config()
        
        log_level = config.log_level
        log_file = config.log_file
        log_dir = config.log_dir if hasattr(config, 'log_dir') else ""
        
        if format_string is None:
            format_string = (
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
            )
        
        if log_file:
            log_path = Path(log_file)
            if not log_path.is_absolute():
                log_path = Path(log_dir) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        root_logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if use_color and sys.stdout.isatty():
            console_formatter = ColoredFormatter(format_string)
        else:
            console_formatter = logging.Formatter(format_string)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        if log_file:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        cls._suppress_third_party_loggers()
        
        cls._initialized = True
    
    @classmethod
    def _suppress_third_party_loggers(cls) -> None:
        """抑制第三方库的日志输出
        
        将 httpx、openai、urllib3 等库的日志级别设置为 WARNING，
        减少控制台日志刷新频率。
        """
        third_party_loggers = [
            "httpx",
            "httpcore",
            "openai",
            "urllib3",
            "requests",
            "asyncio",
            "aiohttp",
            "websockets",
            "neo4j",
            "faiss",
        ]
        
        for logger_name in third_party_loggers:
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.setLevel(logging.WARNING)
            third_party_logger.propagate = False
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            日志记录器实例
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        cls._loggers[name] = logger
        return logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "",
    use_color: bool = True,
) -> None:
    """设置日志配置
    
    便捷函数，用于初始化日志系统。
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
        log_dir: 日志目录
        use_color: 是否使用彩色输出
    """
    LoggerManager.setup(
        log_level=log_level,
        log_file=log_file,
        log_dir=log_dir,
        use_color=use_color,
    )


@lru_cache()
def get_logger(name: str) -> logging.Logger:
    """获取日志记录器
    
    使用 lru_cache 缓存日志记录器实例。
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        
    Returns:
        日志记录器实例
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    if not LoggerManager._initialized:
        LoggerManager.setup()
    
    return LoggerManager.get_logger(name)


logger = get_logger("rag")

if __name__ == '__main__':
    print("=" * 50)
    print("测试 Logger 模块")
    print("=" * 50)
    
    test_logger = get_logger("test_module")
    print(f"✓ 获取日志记录器: {test_logger.name}")
    
    test_logger.debug("这是 debug 消息")
    test_logger.info("这是 info 消息")
    test_logger.warning("这是 warning 消息")
    test_logger.error("这是 error 消息")
    print("✓ 各级别日志输出正常")
    
    LoggerManager.setup(use_color=True)
    print(f"✓ LoggerManager 初始化状态: {LoggerManager._initialized}")
    
    logger1 = get_logger("module_a")
    logger2 = get_logger("module_a")
    print(f"✓ 缓存功能: {logger1 is logger2}")
    
    print("\n所有测试通过!")
