#!/usr/bin/env python3
"""
OpenTrade 备份恢复脚本

功能：
1. 备份配置文件
2. 备份数据库
3. 备份策略数据
4. 一键恢复

Usage:
    python backup.py backup           # 执行备份
    python backup.py restore latest   # 恢复最新备份
    python backup.py list             # 列出备份
    python backup.py clean 7          # 清理7天前的备份
"""

import argparse
import gzip
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

# 配置
BACKUP_DIR = Path("/root/.opentrade/backups")
RETENTION_DAYS = 7
COMPRESSION_LEVEL = 9


def ensure_backup_dir():
    """确保备份目录存在"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    return BACKUP_DIR


def get_timestamp() -> str:
    """获取时间戳"""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def backup_config() -> Path:
    """备份配置文件"""
    config_src = Path.home() / ".opentrade" / "config.yaml"
    if not config_src.exists():
        print("[yellow]⚠️ 配置文件不存在，跳过[/yellow]")
        return None
    
    backup_dir = ensure_backup_dir()
    backup_file = backup_dir / f"config_{get_timestamp()}.yaml.gz"
    
    with open(config_src, 'rb') as f_in:
        with gzip.open(backup_file, 'wb', compresslevel=COMPRESSION_LEVEL) as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"✅ 配置已备份: {backup_file.name}")
    return backup_file


def backup_strategies() -> Path:
    """备份策略数据"""
    data_dir = Path("/root/.opentrade/data")
    if not data_dir.exists():
        print("[yellow]⚠️ 数据目录不存在，跳过[/yellow]")
        return None
    
    backup_dir = ensure_backup_dir()
    backup_file = backup_dir / f"strategies_{get_timestamp()}.json.gz"
    
    # 收集所有策略文件
    strategy_files = list(data_dir.glob("*.json"))
    if not strategy_files:
        print("[yellow]⚠️ 没有策略文件，跳过[/yellow]")
        return None
    
    all_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "files": {},
    }
    
    for file in strategy_files:
        try:
            with open(file, 'r') as f:
                all_data["files"][file.name] = json.load(f)
        except Exception as e:
            print(f"[yellow]⚠️ 读取 {file.name} 失败: {e}[/yellow]")
    
    with gzip.open(backup_file, 'wt', compresslevel=COMPRESSION_LEVEL) as f:
        json.dump(all_data, f, indent=2, default=str)
    
    print(f"✅ 策略已备份: {backup_file.name}")
    return backup_file


def backup_evolution_history() -> Path:
    """备份进化历史"""
    history_file = Path("/root/opentrade/data/evolution_history.json")
    if not history_file.exists():
        # 检查其他可能的位置
        alt_file = Path("/root/.opentrade/data/evolution_history.json")
        if not alt_file.exists():
            print("[yellow]⚠️ 进化历史文件不存在，跳过[/yellow]")
            return None
        history_file = alt_file
    
    backup_dir = ensure_backup_dir()
    backup_file = backup_dir / f"evolution_{get_timestamp()}.json.gz"
    
    with open(history_file, 'rb') as f_in:
        with gzip.open(backup_file, 'wb', compresslevel=COMPRESSION_LEVEL) as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"✅ 进化历史已备份: {backup_file.name}")
    return backup_file


def backup_docker_data() -> Optional[Path]:
    """备份 Docker volumes 数据"""
    docker_dir = BACKUP_DIR / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 备份 PostgreSQL 数据
        result = subprocess.run(
            ["docker", "cp", "opentrade-postgres-1:/var/lib/postgresql", str(docker_dir)],
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✅ PostgreSQL 数据已备份")
        
        # 备份 Redis 数据
        result = subprocess.run(
            ["docker", "cp", "opentrade-redis-1:/data", str(docker_dir / "redis")],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            print("✅ Redis 数据已备份")
        
        # 创建 tar.gz 归档
        tar_file = BACKUP_DIR / f"docker_{get_timestamp()}.tar.gz"
        subprocess.run(
            ["tar", "-czf", str(tar_file), "-C", str(BACKUP_DIR), "docker"],
            check=True
        )
        shutil.rmtree(docker_dir)
        
        print(f"✅ Docker 数据已备份: {tar_file.name}")
        return tar_file
        
    except subprocess.TimeoutExpired:
        print("[red]❌ Docker 备份超时[/red]")
        return None
    except Exception as e:
        print(f"[yellow]⚠️ Docker 备份失败: {e}[/yellow]")
        return None


def backup_all() -> list[Path]:
    """执行完整备份"""
    print("\n[bold cyan]🔒 OpenTrade 备份开始[/bold cyan]")
    print("=" * 50)
    
    backups = []
    
    # 备份配置
    config_backup = backup_config()
    if config_backup:
        backups.append(config_backup)
    
    # 备份策略
    strategy_backup = backup_strategies()
    if strategy_backup:
        backups.append(strategy_backup)
    
    # 备份进化历史
    evolution_backup = backup_evolution_history()
    if evolution_backup:
        backups.append(evolution_backup)
    
    # 统计
    total_size = sum(f.stat().st_size for f in backups) if backups else 0
    print(f"\n📦 备份完成: {len(backups)} 个文件, {total_size / 1024:.1f} KB")
    
    # 清理旧备份
    clean_old_backups()
    
    return backups


def list_backups():
    """列出所有备份"""
    backup_dir = ensure_backup_dir()
    backups = sorted(backup_dir.glob("*.gz"), reverse=True)
    
    if not backups:
        print("\n[yellow]⚠️ 没有找到备份文件[/yellow]")
        return []
    
    print(f"\n[bold]📋 可用备份 ({len(backups)} 个)[/bold]")
    print("-" * 60)
    
    for backup in backups[:10]:  # 只显示最近10个
        size = backup.stat().st_size
        mtime = datetime.fromtimestamp(backup.stat().st_mtime)
        age = datetime.utcnow() - mtime
        
        age_str = f"{age.seconds // 3600}h ago" if age.days == 0 else f"{age.days}d ago"
        
        print(f"  {backup.name:40s} {size/1024:8.1f} KB  {age_str}")
    
    return backups


def clean_old_backups(days: int = None):
    """清理旧备份"""
    if days is None:
        days = RETENTION_DAYS
    
    backup_dir = ensure_backup_dir()
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    removed = 0
    for backup in backup_dir.glob("*.gz"):
        mtime = datetime.fromtimestamp(backup.stat().st_mtime)
        if mtime < cutoff:
            backup.unlink()
            removed += 1
    
    if removed > 0:
        print(f"🧹 清理完成: 删除了 {removed} 个旧备份")


def restore_backup(backup_name: str) -> bool:
    """恢复备份"""
    backup_dir = ensure_backup_dir()
    
    # 查找备份文件
    if backup_name == "latest":
        backups = sorted(backup_dir.glob("*.gz"), reverse=True)
        if not backups:
            print("[red]❌ 没有可用的备份[/red]")
            return False
        backup_file = backups[0]
    else:
        backup_file = backup_dir / backup_name
        if not backup_file.exists():
            # 尝试模糊匹配
            matches = list(backup_dir.glob(f"*{backup_name}*"))
            if matches:
                backup_file = matches[0]
            else:
                print(f"[red]❌ 找不到备份: {backup_name}[/red]")
                return False
    
    print(f"\n[bold cyan]🔓 恢复备份: {backup_file.name}[/bold cyan]")
    
    # 确定备份类型
    if "config" in backup_file.name:
        return restore_config(backup_file)
    elif "strategy" in backup_file.name:
        return restore_strategies(backup_file)
    elif "evolution" in backup_file.name:
        return restore_evolution(backup_file)
    else:
        print("[red]❌ 未知备份类型[/red]")
        return False


def restore_config(backup_file: Path) -> bool:
    """恢复配置"""
    config_dst = Path.home() / ".opentrade" / "config.yaml"
    
    try:
        with gzip.open(backup_file, 'rb') as f:
            content = f.read()
        
        # 备份当前配置
        if config_dst.exists():
            backup_current = config_dst.with_suffix(f".backup_{get_timestamp()}")
            shutil.copy(config_dst, backup_current)
            print(f"📁 当前配置已备份: {backup_current.name}")
        
        with open(config_dst, 'wb') as f:
            f.write(content)
        
        print(f"✅ 配置已恢复: {config_dst}")
        return True
        
    except Exception as e:
        print(f"[red]❌ 恢复配置失败: {e}[/red]")
        return False


def restore_strategies(backup_file: Path) -> bool:
    """恢复策略"""
    data_dir = Path("/root/.opentrade/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with gzip.open(backup_file, 'rt') as f:
            data = json.load(f)
        
        for filename, content in data.get("files", {}).items():
            file_path = data_dir / filename
            
            # 备份当前文件
            if file_path.exists():
                backup = file_path.with_suffix(f".backup_{get_timestamp()}")
                shutil.copy(file_path, backup)
            
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
            
            print(f"✅ 策略已恢复: {filename}")
        
        return True
        
    except Exception as e:
        print(f"[red]❌ 恢复策略失败: {e}[/red]")
        return False


def restore_evolution(backup_file: Path) -> bool:
    """恢复进化历史"""
    history_file = Path("/root/opentrade/data/evolution_history.json")
    
    try:
        with gzip.open(backup_file, 'rb') as f_in:
            content = f_in.read()
        
        # 备份当前
        if history_file.exists():
            backup = history_file.with_suffix(f".backup_{get_timestamp()}")
            shutil.copy(history_file, backup)
        
        with open(history_file, 'wb') as f:
            f.write(content)
        
        print(f"✅ 进化历史已恢复: {history_file}")
        return True
        
    except Exception as e:
        print(f"[red]❌ 恢复进化历史失败: {e}[/red]")
        return False


def main():
    parser = argparse.ArgumentParser(description="OpenTrade 备份恢复工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # backup
    parser_backup = subparsers.add_parser("backup", help="执行备份")
    parser_backup.add_argument("--full", action="store_true", help="包含 Docker 数据")
    
    # restore
    parser_restore = subparsers.add_parser("restore", help="恢复备份")
    parser_restore.add_argument("backup", help="备份文件名或 'latest'")
    
    # list
    subparsers.add_parser("list", help="列出备份")
    
    # clean
    parser_clean = subparsers.add_parser("clean", help="清理旧备份")
    parser_clean.add_argument("days", type=int, nargs="?", default=7, help="保留天数")
    
    args = parser.parse_args()
    
    if args.command == "backup":
        backup_all()
        
    elif args.command == "restore":
        restore_backup(args.backup)
        
    elif args.command == "list":
        list_backups()
        
    elif args.command == "clean":
        clean_old_backups(args.days)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
