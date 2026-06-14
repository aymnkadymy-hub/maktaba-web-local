#!/usr/bin/env bash
# create_keystore.sh — Generate Android release signing keystore
# Run once per machine:  bash scripts/create_keystore.sh
# Output: mobile/keystore/maktaba-release.keystore + mobile/android/key.properties

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KEYSTORE_DIR="mobile/keystore"
KEYSTORE_FILE="$KEYSTORE_DIR/maktaba-release.keystore"
KEY_PROPS="mobile/android/key.properties"

mkdir -p "$KEYSTORE_DIR"

if [ -f "$KEYSTORE_FILE" ]; then
    echo "[INFO] Keystore already exists: $KEYSTORE_FILE"
    echo "       Delete it first if you want to regenerate."
    exit 0
fi

echo ""
echo "==> Generating Android release keystore"
echo "    You will be prompted for passwords and organization info."
echo "    Remember the passwords — they go into key.properties."
echo ""

read -rsp "Enter keystore password (min 6 chars): " STORE_PW; echo
read -rsp "Confirm keystore password: " STORE_PW2; echo
if [ "$STORE_PW" != "$STORE_PW2" ]; then echo "[ERROR] Passwords don't match"; exit 1; fi

read -rsp "Enter key password (or press Enter to use same): " KEY_PW; echo
KEY_PW="${KEY_PW:-$STORE_PW}"

keytool -genkey -v \
    -keystore "$KEYSTORE_FILE" \
    -alias maktaba \
    -keyalg RSA \
    -keysize 2048 \
    -validity 10000 \
    -storepass "$STORE_PW" \
    -keypass "$KEY_PW" \
    -dname "CN=Maktaba App, OU=Mobile, O=Maktaba, L=Unknown, ST=Unknown, C=SA"

echo ""
echo "==> Writing key.properties..."
cat > "$KEY_PROPS" << EOF
storePassword=$STORE_PW
keyPassword=$KEY_PW
keyAlias=maktaba
storeFile=../keystore/maktaba-release.keystore
EOF

chmod 600 "$KEY_PROPS" "$KEYSTORE_FILE"

echo ""
echo "══════════════════════════════════════════════"
echo " Keystore created!"
echo "  File:         $KEYSTORE_FILE"
echo "  key.properties: $KEY_PROPS"
echo ""
echo " IMPORTANT: Back up $KEYSTORE_FILE securely."
echo " If you lose it you cannot publish updates to Play Store."
echo "══════════════════════════════════════════════"
