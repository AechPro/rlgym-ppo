import struct

HEADER_LEN = 3
ENV_SHAPES_HEADER          = [82772., 83273., 83774.]
ENV_RESET_STATE_HEADER     = [83744., 83774., 83876.]
ENV_STEP_DATA_HEADER       = [83775., 53776., 83727.]
POLICY_ACTIONS_HEADER      = [12782., 83783., 80784.]
PROC_MESSAGE_SHAPES_HEADER = [63776., 83777., 83778.]
STOP_MESSAGE_HEADER        = [11781., 83782., 83983.]

def pack_message(message_floats):
    return struct.pack("%sf" % len(message_floats), *message_floats)

def unpack_message(message_bytes):
    return list(struct.unpack('%sf' % (len(message_bytes) // 4), message_bytes))




