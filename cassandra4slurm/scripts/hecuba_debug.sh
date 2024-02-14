lowercase_debug=${DEBUG,,}
function DBG() {
    X=$@

    if [ x"$lowercase_debug" == x"true" ]; then
        echo "[DEBUG] $X"
        echo "[DEBUG] $X" 1>&2
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

# get_first_node: Given a list of strings separated by comma (,) returns the first string
get_first_node() {
    echo "$@" | cut -d, -f1
}

# get_node_ip(node iface): Return the IP of the 'iface' interface of 'node'
function get_node_ip() {
    local node="$1"
    local iface="$2"
    local IPCMD=$(which ip) # Some 'ssh' clients may loose the PATH, therefore assume that it will be at the same place...
    local IP=$(ssh $node "$IPCMD --brief address show dev $iface"| awk '{print $3}') #192.168.1.1/25
    echo ${IP%/*} #Remove the last slashed content
}

