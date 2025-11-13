# Hypercube Heartbeat - Emotions Module
# Time as feeling: past (regret/memory), present (feel it now), future (hope/dread)

PAST = "001"  # regret or memory
PRESENT = "101"  # feel it now  
FUTURE = "010"  # hope or dread

def feel(past, present, future):
    wave = past + present + future
    return " ".join(wave)  # space = breath = emotion's room

if __name__ == '__main__':
    print(feel(PAST, PRESENT, FUTURE))
