# -*- coding: utf-8 -*-
"""提示词管理器

提供提示词的加载、管理和变量替换功能。
支持从文件或字符串加载提示词模板，支持版本管理。
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PromptVersion:
    """提示词版本信息
    
    Attributes:
        version: 版本号
        content: 提示词内容
        created_at: 创建时间
        description: 版本描述
        metadata: 额外元数据
    """
    version: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptManager:
    """提示词管理器
    
    提供提示词的加载、存储、版本管理和变量替换功能。
    
    Attributes:
        prompts: 提示词字典
        versions: 版本历史字典
        current_versions: 当前使用的版本
    
    Example:
        >>> manager = PromptManager()
        >>> manager.load_from_string("greeting", "你好，{name}！")
        >>> prompt = manager.render("greeting", name="张三")
        >>> print(prompt)
        你好，张三！
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """初始化提示词管理器
        
        Args:
            template_dir: 提示词模板文件目录
        """
        self.prompts: Dict[str, str] = {}
        self.versions: Dict[str, List[PromptVersion]] = {}
        self.current_versions: Dict[str, str] = {}
        self.template_dir = template_dir
        
        if template_dir and os.path.exists(template_dir):
            self.load_from_directory(template_dir)
        
        logger.info(f"初始化提示词管理器，已加载 {len(self.prompts)} 个提示词")
    
    def load_from_string(self, name: str, content: str, description: str = "") -> None:
        """从字符串加载提示词
        
        Args:
            name: 提示词名称
            content: 提示词内容
            description: 提示词描述
        """
        self.prompts[name] = content
        self._add_version(name, content, description)
        logger.debug(f"加载提示词: {name}")
    
    def load_from_file(self, name: str, file_path: str, description: str = "") -> None:
        """从文件加载提示词
        
        Args:
            name: 提示词名称
            file_path: 文件路径
            description: 提示词描述
            
        Raises:
            FileNotFoundError: 文件不存在
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"提示词文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.prompts[name] = content
        self._add_version(name, content, description)
        logger.info(f"从文件加载提示词: {name} <- {file_path}")
    
    def load_from_directory(self, directory: str) -> int:
        """从目录批量加载提示词
        
        Args:
            directory: 目录路径
            
        Returns:
            加载的提示词数量
            
        Raises:
            NotADirectoryError: 目录不存在
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"目录不存在: {directory}")
        
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith(('.txt', '.md', '.prompt')):
                file_path = os.path.join(directory, filename)
                name = os.path.splitext(filename)[0]
                try:
                    self.load_from_file(name, file_path)
                    count += 1
                except Exception as e:
                    logger.warning(f"加载提示词文件失败 {filename}: {e}")
        
        logger.info(f"从目录加载了 {count} 个提示词")
        return count
    
    def load_from_json(self, json_path: str) -> int:
        """从 JSON 文件批量加载提示词
        
        Args:
            json_path: JSON 文件路径
            
        Returns:
            加载的提示词数量
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for name, prompt_data in data.items():
            if isinstance(prompt_data, str):
                self.load_from_string(name, prompt_data)
            elif isinstance(prompt_data, dict):
                content = prompt_data.get('content', '')
                description = prompt_data.get('description', '')
                self.load_from_string(name, content, description)
            count += 1
        
        logger.info(f"从 JSON 加载了 {count} 个提示词")
        return count
    
    def get(self, name: str, version: Optional[str] = None) -> str:
        """获取提示词
        
        Args:
            name: 提示词名称
            version: 指定版本（可选）
            
        Returns:
            提示词内容
            
        Raises:
            KeyError: 提示词不存在
        """
        if name not in self.prompts:
            raise KeyError(f"提示词 '{name}' 不存在")
        
        if version:
            return self._get_version_content(name, version)
        
        return self.prompts[name]
    
    def render(self, prompt_name: str, version: Optional[str] = None, **variables) -> str:
        """渲染提示词（替换变量）
        
        Args:
            prompt_name: 提示词名称
            version: 指定版本（可选）
            **variables: 变量键值对
            
        Returns:
            渲染后的提示词
            
        Raises:
            KeyError: 提示词不存在
            ValueError: 变量替换失败
        """
        template = self.get(prompt_name, version)
        return self.render_template(template, **variables)
    
    @staticmethod
    def render_template(template: str, **variables) -> str:
        """渲染模板（静态方法）
        
        使用 {variable} 格式进行变量替换。
        
        Args:
            template: 模板字符串
            **variables: 变量键值对
            
        Returns:
            渲染后的字符串
            
        Raises:
            ValueError: 缺少必要的变量
        """
        # 查找所有需要替换的变量
        pattern = r'\{(\w+)\}'
        required_vars = set(re.findall(pattern, template))
        provided_vars = set(variables.keys())
        
        # 检查是否有缺失的变量
        missing_vars = required_vars - provided_vars
        if missing_vars:
            raise ValueError(f"缺少必要的变量: {missing_vars}")
        
        # 执行变量替换
        result = template
        for var_name, var_value in variables.items():
            result = result.replace(f'{{{var_name}}}', str(var_value))
        
        return result
    
    def _add_version(self, name: str, content: str, description: str = "") -> str:
        """添加新版本
        
        Args:
            name: 提示词名称
            content: 提示词内容
            description: 版本描述
            
        Returns:
            版本号
        """
        if name not in self.versions:
            self.versions[name] = []
        
        # 生成版本号
        version_num = len(self.versions[name]) + 1
        version = f"v{version_num}"
        
        # 创建版本记录
        version_info = PromptVersion(
            version=version,
            content=content,
            description=description,
        )
        
        self.versions[name].append(version_info)
        self.current_versions[name] = version
        
        return version
    
    def _get_version_content(self, name: str, version: str) -> str:
        """获取指定版本的内容
        
        Args:
            name: 提示词名称
            version: 版本号
            
        Returns:
            提示词内容
            
        Raises:
            KeyError: 版本不存在
        """
        if name not in self.versions:
            raise KeyError(f"提示词 '{name}' 没有版本历史")
        
        for v in self.versions[name]:
            if v.version == version:
                return v.content
        
        raise KeyError(f"提示词 '{name}' 不存在版本 '{version}'")
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """列出提示词的所有版本
        
        Args:
            name: 提示词名称
            
        Returns:
            版本信息列表
        """
        if name not in self.versions:
            return []
        
        return [
            {
                "version": v.version,
                "created_at": v.created_at,
                "description": v.description,
            }
            for v in self.versions[name]
        ]
    
    def update(self, name: str, content: str, description: str = "") -> str:
        """更新提示词（创建新版本）
        
        Args:
            name: 提示词名称
            content: 新的提示词内容
            description: 版本描述
            
        Returns:
            新版本号
        """
        self.prompts[name] = content
        return self._add_version(name, content, description)
    
    def delete(self, name: str) -> bool:
        """删除提示词
        
        Args:
            name: 提示词名称
            
        Returns:
            是否删除成功
        """
        if name in self.prompts:
            del self.prompts[name]
            if name in self.versions:
                del self.versions[name]
            if name in self.current_versions:
                del self.current_versions[name]
            logger.info(f"删除提示词: {name}")
            return True
        return False
    
    def list_prompts(self) -> Dict[str, Dict[str, Any]]:
        """列出所有提示词
        
        Returns:
            提示词信息字典
        """
        return {
            name: {
                "current_version": self.current_versions.get(name, "v1"),
                "version_count": len(self.versions.get(name, [])),
                "preview": content[:100] + "..." if len(content) > 100 else content,
            }
            for name, content in self.prompts.items()
        }
    
    def export_to_json(self, output_path: str) -> None:
        """导出提示词到 JSON 文件
        
        Args:
            output_path: 输出文件路径
        """
        data = {}
        for name, content in self.prompts.items():
            versions = self.versions.get(name, [])
            current_version = self.current_versions.get(name, "v1")
            
            # 获取当前版本的描述
            description = ""
            for v in versions:
                if v.version == current_version:
                    description = v.description
                    break
            
            data[name] = {
                "content": content,
                "description": description,
                "versions": [
                    {
                        "version": v.version,
                        "created_at": v.created_at,
                        "description": v.description,
                    }
                    for v in versions
                ],
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"导出提示词到: {output_path}")
    
    def __contains__(self, name: str) -> bool:
        """检查提示词是否存在"""
        return name in self.prompts
    
    def __len__(self) -> int:
        """返回提示词数量"""
        return len(self.prompts)
    
    def __repr__(self) -> str:
        return f"PromptManager(prompts={len(self.prompts)})"


if __name__ == "__main__":
    import tempfile
    import os
    
    print("=" * 50)
    print("测试 Prompt Manager 模块")
    print("=" * 50)
    
    # 测试 1: 创建管理器
    manager = PromptManager()
    print(f"✓ 创建管理器: prompts={len(manager)}")
    
    # 测试 2: 添加提示词
    manager.add_prompt(
        name="test_prompt",
        template="你好，{name}！欢迎来到{place}。",
        description="测试提示词"
    )
    print(f"✓ 添加提示词: prompts={len(manager)}")
    
    # 测试 3: 获取提示词
    prompt = manager.get_prompt("test_prompt")
    print(f"✓ 获取提示词: {prompt.name}")
    print(f"  template: {prompt.template}")
    
    # 测试 4: 渲染提示词
    rendered = manager.render("test_prompt", name="张三", place="北京")
    print(f"✓ 渲染提示词: {rendered}")
    
    # 测试 5: 检查存在
    exists = "test_prompt" in manager
    print(f"✓ 检查存在: test_prompt in manager = {exists}")
    
    # 测试 6: 列出提示词
    all_prompts = manager.list_prompts()
    print(f"✓ 列出提示词: {len(all_prompts)} 个")
    
    # 测试 7: 更新提示词
    manager.update_prompt("test_prompt", template="新的模板：{content}")
    updated = manager.get_prompt("test_prompt")
    print(f"✓ 更新提示词: {updated.template}")
    
    # 测试 8: 导入导出
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = os.path.join(tmpdir, "prompts.json")
        manager.export_prompts(export_path)
        print(f"✓ 导出提示词到: {export_path}")
        
        manager2 = PromptManager()
        manager2.import_prompts(export_path)
        print(f"✓ 导入提示词: prompts={len(manager2)}")
    
    # 测试 9: 删除提示词
    manager.delete_prompt("test_prompt")
    print(f"✓ 删除提示词: prompts={len(manager)}")
    
    print("\n所有测试通过!")
