"""镜像同步工具"""
# pylint: disable=broad-exception-caught,import-outside-toplevel

import sys
import traceback
import uuid
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
        visibility: bool | None = False,
        retry: int | None = 3,
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
            visibility (bool | None): 当目标仓库不存在时自动创建的仓库可见性.
            retry (int | None): 上传重试次数.
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

        if not self.repo_manager.check_repo(
            api_type=dst_repo,
            repo_id=dst_repo_id,
            repo_type=dst_repo_type,
            visibility=visibility,
        ):
            logger.error(
                "检查 %s/%s (类型: %s) 仓库失败, 无法镜像仓库",
                dst_repo,
                dst_repo_id,
                dst_repo_type,
            )
            return

        self.make_hf_or_ms_repo_mirror(
            src_repo=src_repo,
            dst_repo=dst_repo,
            src_repo_id=src_repo_id,
            dst_repo_id=dst_repo_id,
            src_repo_type=src_repo_type,
            dst_repo_type=dst_repo_type,
            retry=retry,
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
        retry: int | None = 3,
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
            retry (int | None): 上传重试次数.
        """
        from tqdm import tqdm
        from modelscope import snapshot_download

        src_repo_files = set(
            self.repo_manager.get_repo_file(
                api_type=src_repo,
                repo_id=src_repo_id,
                repo_type=src_repo_type,
            )
        )
        dst_repo_files = set(
            self.repo_manager.get_repo_file(
                api_type=dst_repo,
                repo_id=dst_repo_id,
                repo_type=dst_repo_type,
            )
        )
        need_sync_files = [
            x
            for x in tqdm(src_repo_files, desc=f"统计需要镜像到 {dst_repo} 的文件")
            if x not in dst_repo_files
        ]
        files_count = len(need_sync_files)
        logger.info("需要镜像的文件数量: %s", files_count)
        count = 0
        retry_sum = 0
        tmp_dir = self.workspace / f"{uuid.uuid4()}"
        for file in need_sync_files:
            count += 1
            logger.info(
                "[%s/%s] 镜像 %s 到 %s (类型: %s) 中",
                count,
                files_count,
                file,
                dst_repo_id,
                dst_repo_type,
            )
            while retry_sum < retry:
                try:
                    if src_repo == "huggingface":
                        self.repo_manager.hf_api.hf_hub_download(
                            repo_id=src_repo_id,
                            repo_type=src_repo_type,
                            filename=file,
                            local_dir=tmp_dir,
                        )
                    elif src_repo == "modelscope":
                        snapshot_download(
                            repo_id=src_repo_id,
                            repo_type=src_repo_type,
                            allow_patterns=file,
                            local_dir=tmp_dir,
                        )
                    file_path = tmp_dir / file
                    if dst_repo == "huggingface":
                        self.repo_manager.hf_api.upload_file(
                            path_or_fileobj=file_path,
                            path_in_repo=file,
                            repo_id=dst_repo_id,
                            repo_type=dst_repo_type,
                            commit_message=f"Upload {file}",
                        )
                    elif dst_repo == "modelscope":
                        self.repo_manager.ms_api.upload_file(
                            path_or_fileobj=file_path,
                            path_in_repo=file,
                            repo_id=dst_repo_id,
                            repo_type=dst_repo_type,
                            commit_message=f"Upload {file}",
                            token=self.repo_manager.ms_token,
                        )
                    self.remove_files(file_path)
                    break
                except Exception as e:
                    traceback.print_exc()
                    logger.error(
                        "[%s/%s] 镜像 %s 时发生了错误: %s", count, files_count, file, e
                    )
                    if retry_sum < retry:
                        logger.warning("重新镜像 %s 中", file)
        logger.info("镜像仓库完成")
        if tmp_dir.exists():
            self.remove_files(tmp_dir)

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
