lowercase_debug=${DEBUG,,}
function DBG() {
    X=$@

    if [ x"$lowercase_debug" == x"true" ]; then
        echo "[DEBUG] $X"
    fi
}
