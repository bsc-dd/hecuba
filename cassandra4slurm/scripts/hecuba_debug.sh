lowercase_debug=${DEBUG,,}
function DBG() {
    X=$@

    if [ x"$lowercase_debug" == x"true" ]; then
        echo "[DEBUG] $X"
    fi
}

# show_time MSG START_TIME END_TIME
show_time () {
    local MSG="$1"
    local RECOVERTIME1="$2"
    local RECOVERTIME2="$3"

    local MILL1=$(echo $RECOVERTIME1 | cut -c 10-12)
    local MILL2=$(echo $RECOVERTIME2 | cut -c 10-12)
    local TIMESEC1=$(date -d "$RECOVERTIME1" +%s)
    local TIMESEC2=$(date -d "$RECOVERTIME2" +%s)
    local TIMESEC=$(( 10#$TIMESEC2 - 10#$TIMESEC1 ))
    local MILL=$(( 10#$MILL2 - 10#$MILL1 ))

    # Adjusting seconds if necessary
    if [ $MILL -lt 0 ]
    then
        MILL=$(( 1000 + 10#$MILL ))
        TIMESEC=$(( 10#$TIMESEC - 1 ))
    fi

    echo "$MSG ${TIMESEC}s.  ${MILL}ms."
}

die() {
    echo "$@"
    exit
}

run() {
    X="$@"
    DBG "$X"
    eval "$X"
}
