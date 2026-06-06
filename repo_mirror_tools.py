"""镜像同步工具"""
# pylint: disable=broad-exception-caught,import-outside-toplevel

import sys
from typing import Literal
from pathlib import Path

from sd_webui_all_in_one import logger, VERSION, BaseManager  # type: ignore
from sd_webui_all_in_one.utils import clear_jupyter_output  # type: ignore
from sd_webui_all_in_one.env_manager import configure_pip  # type: ignore
from sd_webui_all_in_one.pkg_manager import install_manager_depend  # type: ignore


class RepoMirrorTools(BaseManager):
    """
    镜像制作工具

    提供镜像仓库管理、缓存清理、令牌验证以及环境安装等功能.
    """

    def clean_package_cache(self) -> bool:
        """
        清理 APT、pip 和 uv 的缓存.

        Returns:
            bool: 清理成功时返回 True, 否则返回 False.
        """
        logger.info("清理缓存中")
        try:
            self.run_cmd(
                [Path(sys.executable).as_posix(), "-m", "pip", "cache", "purge"]
            )
            self.run_cmd(["uv", "cache", "clean"])
            self.run_cmd(["apt", "clean"])
            logger.info("缓存清理完成")
            return True
        except Exception as e:
            logger.error("清理缓存时出现错误: %s", e)
            return False

    def verify_huggingface_token(self, hf_token: str) -> bool:
        """
        验证 HuggingFace 访问令牌.

        Args:
            hf_token (str): HuggingFace 账号 Token.

        Returns:
            bool: 验证成功时返回 True, 否则返回 False.
        """
        from huggingface_hub import HfApi

        logger.info("验证 HuggingFace Token 中")
        api = HfApi()
        try:
            api.whoami(hf_token)
            logger.info("HuggingFace Token 验证成功")
            return True
        except Exception as e:
            logger.error("HuggingFace Token 验证失败: %s", e)
            return False

    def verify_modelscope_token(self, ms_token: str) -> bool:
        """
        验证 ModelScope 访问令牌.

        Args:
            ms_token (str): ModelScope 账号 Token.

        Returns:
            bool: 验证成功时返回 True, 否则返回 False.
        """
        from modelscope import HubApi

        api = HubApi()
        logger.info("验证 ModelScope Token 中")
        try:
            api.login(ms_token)
            logger.info("ModelScope Token 验证成功")
            return True
        except Exception as e:
            logger.error("ModelScope Token 验证失败: %s", e)
            return False

    def generate_repo_url(
        self,
        api_type: Literal["huggingface", "modelscope"],
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"],
    ) -> str | None:
        """
        生成 HuggingFace 或 ModelScope 仓库访问地址.

        Args:
            api_type (Literal["huggingface", "modelscope"]): API 类型.
            repo_id (str): 仓库 ID.
            repo_type (Literal["model", "dataset", "space"]): 仓库类型.

        Returns:
            str | None: 仓库访问地址, 若 api_type 不支持则返回 None.
        """

        if api_type == "huggingface":
            if repo_type == "model":
                return f"https://huggingface.co/{repo_id}"
            if repo_type == "dataset":
                return f"https://huggingface.co/datasets/{repo_id}"
            if repo_type == "space":
                return f"https://huggingface.co/spaces/{repo_id}"
        elif api_type == "modelscope":
            if repo_type == "model":
                return f"https://modelscope.cn/models/{repo_id}"
            if repo_type == "dataset":
                return f"https://modelscope.cn/datasets/{repo_id}"
            if repo_type == "space":
                return f"https://modelscope.cn/studios/{repo_id}"
        else:
            logger.error("未知的 Api 类型: %s", api_type)
            return None

    def sync_repo(
        self,
        src_repo: Literal["huggingface", "modelscope"],
        dst_repo: Literal["huggingface", "modelscope"],
        src_repo_id: str,
        dst_repo_id: str,
        src_repo_type: Literal["model", "dataset", "space"] = "model",
        dst_repo_type: Literal["model", "dataset", "space"] = "model",
        visibility: bool = False,
        revision: str | None = None,
        retry: int = 3,
        max_workers: int = 1,
        use_fast_download: bool = False,
        download_tool: Literal["aria2", "requests", "urllib"] | None = "requests",
        download_num_threads: int = 8,
        download_progress: bool = True,
    ) -> None:
        """
        镜像 HuggingFace / ModelScope 仓库.

        Args:
            src_repo (Literal["huggingface", "modelscope"]): 源仓库类型.
            dst_repo (Literal["huggingface", "modelscope"]): 目标仓库类型.
            src_repo_id (str): 源仓库 ID.
            dst_repo_id (str): 目标仓库 ID.
            src_repo_type (Literal["model", "dataset", "space"]): 源仓库类型.
            dst_repo_type (Literal["model", "dataset", "space"]): 目标仓库类型.
            visibility (bool): 当目标仓库不存在时自动创建的仓库可见性.
            revision (str | None): 指定仓库分支、标签或提交哈希.
            retry (int): 单个文件镜像失败后的重试次数.
            max_workers (int): 同步使用的线程数.
            use_fast_download (bool): 是否使用 sd-webui-all-in-one 下载器高速下载.
            download_tool (Literal["aria2", "requests", "urllib"] | None): 高速下载使用的下载器.
            download_num_threads (int): 高速下载线程数.
            download_progress (bool): 高速下载时是否显示下载进度.
        """
        if src_repo not in ["huggingface", "modelscope"]:
            logger.error("未知的镜像仓库类型: %s", src_repo)
            return
        if dst_repo not in ["huggingface", "modelscope"]:
            logger.error("未知的镜像仓库类型: %s", dst_repo)
            return

        logger.info(
            "镜像仓库: %s/%s -> %s/%s", src_repo, src_repo_id, dst_repo, dst_repo_id
        )

        self.mirror_repo_files(
            src_api_type=src_repo,
            dst_api_type=dst_repo,
            src_repo_id=src_repo_id,
            dst_repo_id=dst_repo_id,
            src_repo_type=src_repo_type,
            dst_repo_type=dst_repo_type,
            visibility=visibility,
            revision=revision,
            num_threads=max_workers,
            retry_times=retry,
            use_fast_download=use_fast_download,
            download_tool=download_tool,
            download_num_threads=download_num_threads,
            download_progress=download_progress,
        )
        src_repo_url = self.generate_repo_url(
            api_type=src_repo,
            repo_id=src_repo_id,
            repo_type=src_repo_type,
        )
        dst_repo_url = self.generate_repo_url(
            api_type=dst_repo,
            repo_id=dst_repo_id,
            repo_type=dst_repo_type,
        )
        logger.info("%s -> %s", src_repo_url, dst_repo_url)

    def make_hf_or_ms_repo_mirror(
        self,
        src_repo: Literal["huggingface", "modelscope"],
        dst_repo: Literal["huggingface", "modelscope"],
        src_repo_id: str,
        dst_repo_id: str,
        src_repo_type: Literal["model", "dataset", "space"] = "model",
        dst_repo_type: Literal["model", "dataset", "space"] = "model",
        retry: int = 3,
        max_workers: int = 1,
        revision: str | None = None,
        use_fast_download: bool = False,
        download_tool: Literal["aria2", "requests", "urllib"] | None = "requests",
        download_num_threads: int = 8,
        download_progress: bool = True,
    ) -> None:
        """
        镜像 HuggingFace / ModelScope 仓库文件.

        Args:
            src_repo (Literal["huggingface", "modelscope"]): 源仓库类型.
            dst_repo (Literal["huggingface", "modelscope"]): 目标仓库类型.
            src_repo_id (str): 源仓库 ID.
            dst_repo_id (str): 目标仓库 ID.
            src_repo_type (Literal["model", "dataset", "space"]): 源仓库类型.
            dst_repo_type (Literal["model", "dataset", "space"]): 目标仓库类型.
            retry (int): 单个文件镜像失败后的重试次数.
            max_workers (int): 同步使用的线程数.
            revision (str | None): 指定仓库分支、标签或提交哈希.
            use_fast_download (bool): 是否使用 sd-webui-all-in-one 下载器高速下载.
            download_tool (Literal["aria2", "requests", "urllib"] | None): 高速下载使用的下载器.
            download_num_threads (int): 高速下载线程数.
            download_progress (bool): 高速下载时是否显示下载进度.
        """
        self.mirror_repo_files(
            src_api_type=src_repo,
            dst_api_type=dst_repo,
            src_repo_id=src_repo_id,
            dst_repo_id=dst_repo_id,
            src_repo_type=src_repo_type,
            dst_repo_type=dst_repo_type,
            revision=revision,
            num_threads=max_workers,
            retry_times=retry,
            use_fast_download=use_fast_download,
            download_tool=download_tool,
            download_num_threads=download_num_threads,
            download_progress=download_progress,
        )

    def install(
        self,
        use_uv: bool | None = True,
        huggingface_token: str | None = None,
        modelscope_token: str | None = None,
        clean_install_log: bool | None = False,
    ) -> None:
        """
        安装并配置镜像制作工具所需的运行环境.

        Args:
            use_uv (bool | None): 是否使用 uv 命令进行依赖安装, 默认 True.
            huggingface_token (str | None): HuggingFace 账号 Token, 用于验证和登录.
            modelscope_token (str | None): ModelScope 账号 Token, 用于验证和登录.
            clean_install_log (bool | None): 是否清理安装日志输出, 默认 False.
        """
        logger.info("配置镜像制作工具环境中")
        configure_pip()
        install_manager_depend(use_uv)
        self.clean_package_cache()
        if clean_install_log:
            clear_jupyter_output()
        if huggingface_token is not None and not self.verify_huggingface_token(
            huggingface_token
        ):
            logger.warning("请检查 HuggingFace Token 是否可用")
        if modelscope_token is not None and not self.verify_modelscope_token(
            modelscope_token
        ):
            logger.warning("请检查 ModelScope Token 是否可用")
        self.restart_repo_manager(
            hf_token=huggingface_token,
            ms_token=modelscope_token,
        )
        logger.info("配置镜像制作工具环境完成")


__all__ = [
    "logger",
    "VERSION",
    "RepoMirrorTools",
]
