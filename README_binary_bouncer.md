Binary Bouncer DESCRIPTION:
This tool is run on an ELF or PE binary and tries to return 2 buckets:

    1) ML bucket -- result of Unix strings utility with all the ascii gobbledygook filtered out through running a trained LSTM model on the data
    
    2) Import bucket -->
                        * PEs: list of import/export symbols
                        * ELFs: dump of .strtab

    RUNNING:
        * Requirements: Existence of /buckets & /model folders within project directory   (nothing needed in it)
        * You must run with the '-t' or '--train' flag to create model.
        * '--help' for help
	* readelf (Unix tool) available to system preferable (less in Import bucket w/o)

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
    
    python3 Binary_Bouncer.py [OPTIONS] INCOMING_BINARY
    or
    1) chmod +x Binary_Bouncer.py  (one time, turning into executable)
    2) --> ./Binary_Bouncer.py [OPTIONS] INCOMING_BINARY

    Options:
    -t, --train    set this flag if model needs to be trained
    -e, --enable   set this flag to enable automatic validation testing of trained LSTM model
    -s, --seed     set this flag to seed pyTorch and random computations (if you need an exact reproducibility); normally unnecessary
    --help         Show this message and exit.

