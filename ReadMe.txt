Programming Language: Python 3.9.7
Environment: Anaconda 4.11.0

Libraries Needed:
  numpy
  pandas
  glob
  nltk
  glob
  re
  os
  sys
  math
  pathlib
  collections
  pickle
  warnings

Steps to execute:

Step 1: Make sure you install all the libraries mentioned above before running the assignment.
Step 2: Open the terminal window in the part 4 folder. 
Step 3: Issue the command in this format, write: python3 Q4.py Input.txt
	Input.txt will be any file in txt or csv format having 40 queries
Step 4: The output of all the three IR Systems should be present in the "Part 4" folder in the format
        as specified in the question. Their names would be:
        1. BM25_output.csv        
        2. BRS_output.csv         
        3. TfIdf_output.csv

NOTE: 
      1. The entire preprocessing is already performed on the ~8351 text files and the output files
      are saved in the folder "Data files required" as a pickle format. The corresponding code is contained in "Part 1 and 2" folder.   If you wish 
      to recreate the processed documents, please go to code "Q1 and Q2" and comment break statements with written "comment this" on them. HOWEVER, THIS WILL TAKE
      AROUND ~60-80 MINUTES.

      2.Upload corpus folder "englist corpora" in folder "Data files required" before recreating files from step 1. The corpus is not included due to its huge size. If you want to do the preprocessing as mentioned above,
      please download the corpus from "https://www.cse.iitk.ac.in/users/arnabb/ir/english/" and extract it. At the end there should be a folder named "english-corpora" containing all the raw text files.

     
  
