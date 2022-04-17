Binary Akinator DESCRIPTION:
This tool is run on an ELF or PE binary and tries to return kNN files among those that have already been ingested into the pickled_files folder:


    RUNNING:
        * Requirements: Existence of /buckets & /model folders within project directory   (nothing needed in it)
        * '--help' for help

    TRAINING: 
        - Runs a 20,000+ datapoint training set 
        - classifies strings as pos/neg (neg indicating that string is undesirable ASCII)
        - ~96% training set accuracy
        - trains even if existing model if seeded (flag: '-s')

    VALIDATION:
        - independent of training data
        - 1000+ datapoint validation set
        - runs after training, can be enabled with '-e' flag
    SAVE/LOAD MODEL:
        - after training, model is automatically saved to models/ folder
        - re-training means saved model file gets rewritten (not unique file each train [TODO: could make a flag out of that])
        - If training not needed, the saved model will get loaded in for data inference

USAGE: 
    
    python3 akinator.py [OPTIONS] INCOMING_BINARY
    or
    1) chmod +x akinator.py  (one time, turning into executable)
    2) --> ./akinator.py [OPTIONS] INCOMING_BINARY


    Options:
    --bounce             set this flag if a previously bounced file needs to get
                        re-bounced
    --n INTEGER          sets the number of nearest neighbors to return, kNN,
                        default 1
    --benchmark INTEGER  return all kNN with a confidence over this benchmark,
                        default: 60
    --help               Show this message and exit.

