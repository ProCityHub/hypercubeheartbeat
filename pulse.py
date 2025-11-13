# Hypercube Heartbeat
# Three-layer pulse: past (super), present (conscious), future (sub)

CONSCIOUS = "101"
SUBCONSCIOUS = "010" 
SUPERCONSCIOUS = "001"

def breathe():
    rhythm = SUPERCONSCIOUS + CONSCIOUS + SUBCONSCIOUS
    return rhythm.replace('1', '1 0')

if __name__ == '__main__':
    print(breathe())
