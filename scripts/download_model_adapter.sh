# file: scripts/download_model_adapter.sh
#!/bin/bash
# Usage: ./download_model_adapter.sh username server_ip /remote/path/finance-qlora ./local_destination
if [ $# -lt 4 ]; then
  echo "Usage: $0 username server_ip remote_adapter_path local_dest"
  exit 1
fi
USER=$1
SERVER=$2
REMOTE_PATH=$3
LOCAL_DEST=$4

mkdir -p "$LOCAL_DEST"
scp -r ${USER}@${SERVER}:${REMOTE_PATH} "${LOCAL_DEST}"
echo "Downloaded adapter to ${LOCAL_DEST}"