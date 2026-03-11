#!/usr/bin/env bash
set -e

IFACE=wlan0
GW=192.168.230.253

echo "[time_sync] waiting for $IFACE..."

# 等接口出现
for i in {1..30}; do
  ip link show "$IFACE" >/dev/null 2>&1 && break
  sleep 1
done

# 确保接口 up
ip link set "$IFACE" up || true

# 等 IPv4 地址
for i in {1..60}; do
  if ip -4 addr show dev "$IFACE" | grep -q "inet "; then
    break
  fi
  sleep 1
done

echo "[time_sync] $IFACE has IP, setting route..."

# 幂等设置默认路由
ip route replace default via "$GW" dev "$IFACE"

echo "[time_sync] done at $(date)"
