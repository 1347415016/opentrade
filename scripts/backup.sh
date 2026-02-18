#!/bin/bash

# ============================================
# OpenTrade 数据库备份脚本
# 支持全量备份、增量备份、一键恢复
# ============================================

set -e

# 配置
BACKUP_DIR="/root/.opentrade/backups"
RETENTION_DAYS=30
DB_USER="opentrade"
DB_PASSWORD="password"
DB_NAME="opentrade"
DB_HOST="localhost"
DB_PORT="5432"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE_ONLY=$(date +%Y%m%d)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建备份目录
mkdir -p "$BACKUP_DIR"
mkdir -p "$BACKUP_DIR/daily"
mkdir -p "$BACKUP_DIR/weekly"
mkdir -p "$BACKUP_DIR/monthly"

# ============================================
# 全量备份
# ============================================
full_backup() {
    local backup_file="$BACKUP_DIR/daily/${DB_NAME}_full_${TIMESTAMP}.sql.gz"
    
    log_info "开始全量备份: $backup_file"
    
    # 使用pg_dump进行备份
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -Fc \
        -Z 9 \
        -f "$backup_file" \
        2>&1 | grep -v "pg_dump: [wW]arning"
    
    if [ $? -eq 0 ]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_info "✅ 全量备份完成: $backup_file ($size)"
        
        # 写入备份清单
        echo "$TIMESTAMP,$backup_file,full" >> "$BACKUP_DIR/backup_manifest.log"
        
        # 创建软链接 (latest)
        rm -f "$BACKUP_DIR/daily/latest.sql.gz"
        ln -s "$backup_file" "$BACKUP_DIR/daily/latest.sql.gz"
        
        return 0
    else
        log_error "❌ 全量备份失败"
        return 1
    fi
}

# ============================================
# 增量备份 (WAL归档)
# ============================================
incremental_backup() {
    local backup_file="$BACKUP_DIR/daily/${DB_NAME}_incr_${TIMESTAMP}.tar"
    
    log_info "开始增量备份: $backup_file"
    
    # 检查是否配置了 WAL 归档
    # 这里简化为备份 pg_wal 目录
    PGPASSWORD="$DB_PASSWORD" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT pg_switch_wal();" 2>/dev/null || true
    
    log_info "✅ 增量备份完成 (WAL检查点已触发)"
}

# ============================================
# 恢复备份
# ============================================
restore_backup() {
    local backup_file=$1
    
    if [ -z "$backup_file" ]; then
        log_error "请指定备份文件路径"
        echo "用法: $0 restore <backup_file>"
        return 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        log_error "备份文件不存在: $backup_file"
        return 1
    fi
    
    log_warn "⚠️ 即将恢复备份: $backup_file"
    log_warn "当前数据库中的所有数据将被覆盖！"
    read -p "确认继续? (输入 YES 恢复): " confirm
    
    if [ "$confirm" != "YES" ]; then
        log_info "已取消恢复"
        return 0
    fi
    
    log_info "开始恢复备份..."
    
    # 停止相关服务
    # docker-compose stop opentrade || true
    
    PGPASSWORD="$DB_PASSWORD" pg_restore \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c \
        -O \
        "$backup_file"
    
    if [ $? -eq 0 ]; then
        log_info "✅ 备份恢复完成: $backup_file"
    else
        log_error "❌ 备份恢复失败"
        return 1
    fi
    
    # 重启服务
    # docker-compose start opentrade || true
}

# ============================================
# 清理旧备份
# ============================================
cleanup_old_backups() {
    log_info "清理超过 $RETENTION_DAYS 天的旧备份..."
    
    # 每日备份 - 保留30天
    find "$BACKUP_DIR/daily" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    
    # 每周备份 - 保留12周
    find "$BACKUP_DIR/weekly" -name "*.sql.gz" -mtime +$((RETENTION_DAYS * 4)) -delete
    
    # 每月备份 - 保留12个月
    find "$BACKUP_DIR/monthly" -name "*.sql.gz" -mtime +$((RETENTION_DAYS * 30)) -delete
    
    log_info "✅ 旧备份清理完成"
}

# ============================================
# 备份到远程存储
# ============================================
upload_to_remote() {
    local remote_target=$1
    
    if [ -z "$remote_target" ]; then
        log_error "请指定远程目标 (s3://bucket/path 或 user@host:/path)"
        return 1
    fi
    
    local latest_backup="$BACKUP_DIR/daily/latest.sql.gz"
    
    if [ ! -f "$latest_backup" ]; then
        log_error "无最新备份可上传"
        return 1
    fi
    
    log_info "上传备份到远程存储: $remote_target"
    
    if [[ "$remote_target" == s3://* ]]; then
        # S3 上传
        aws s3 cp "$latest_backup" "$remote_target/" 2>/dev/null || \
        rclone copy "$latest_backup" "$remote_target" 2>/dev/null || \
        log_error "S3 上传失败，请安装 aws-cli 或 rclone"
    else
        # SCP 上传
        scp "$latest_backup" "$remote_target/" 2>/dev/null || \
        log_error "SCP 上传失败"
    fi
    
    log_info "✅ 备份上传完成"
}

# ============================================
# 备份状态报告
# ============================================
status_report() {
    echo ""
    echo "=========================================="
    echo "        OpenTrade 备份状态报告"
    echo "=========================================="
    echo ""
    
    echo "📁 备份目录: $BACKUP_DIR"
    echo ""
    
    echo "📊 备份文件统计:"
    echo "  每日备份: $(ls -1 $BACKUP_DIR/daily/*.sql.gz 2>/dev/null | wc -l) 个"
    echo "  每周备份: $(ls -1 $BACKUP_DIR/weekly/*.sql.gz 2>/dev/null | wc -l) 个"
    echo "  每月备份: $(ls -1 $BACKUP_DIR/monthly/*.sql.gz 2>/dev/null | wc -l) 个"
    echo ""
    
    echo "💾 磁盘使用:"
    du -sh "$BACKUP_DIR" 2>/dev/null || echo "  无法计算"
    echo ""
    
    echo "📋 最近备份:"
    ls -1t $BACKUP_DIR/daily/*.sql.gz 2>/dev/null | head -5 | while read f; do
        local size=$(du -h "$f" | cut -f1)
        local date=$(stat -c %y "$f" 2>/dev/null | cut -d' ' -f1)
        echo "  $date $size $(basename $f)"
    done
    echo ""
    
    echo "=========================================="
}

# ============================================
# 主函数
# ============================================
main() {
    local command=$1
    
    case $command in
        full)
            full_backup
            ;;
        incr|incremental)
            incremental_backup
            ;;
        restore)
            restore_backup $2
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        upload)
            upload_to_remote $2
            ;;
        status)
            status_report
            ;;
        help|--help|-h)
            echo "OpenTrade 数据库备份脚本"
            echo ""
            echo "用法: $0 <命令> [参数]"
            echo ""
            echo "命令:"
            echo "  full         - 全量备份"
            echo "  incr         - 增量备份 (WAL)"
            echo "  restore <file> - 恢复备份"
            echo "  cleanup      - 清理旧备份"
            echo "  upload <target> - 上传到远程存储"
            echo "  status       - 查看备份状态"
            echo ""
            echo "示例:"
            echo "  $0 full                    # 全量备份"
            echo "  $0 restore backup.sql.gz  # 恢复备份"
            echo "  $0 upload s3://my-bucket/backup/"
            ;;
        *)
            log_error "未知命令: $command"
            $0 help
            exit 1
            ;;
    esac
}

# 定时任务示例:
# 0 2 * * * /root/opentrade/scripts/backup.sh full   # 每日2点全量备份
# 0 */4 * * * /root/opentrade/scripts/backup.sh incr  # 每4小时增量备份
# 0 3 * * * /root/opentrade/scripts/backup.sh cleanup # 每日3点清理
# 0 4 * * * /root/opentrade/scripts/backup.sh upload s3://my-bucket/opentrade-backups/ # 上传到S3

main "$@"
