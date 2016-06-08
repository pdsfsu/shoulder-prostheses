# shoulder-prostheses
To run segmentation and circle detection tests:
#### python ./segmentation.py img_file_dir/ [names text doc] [circle only? True/False] [threshold]
  
  - img_file_dir: the directory containing the image files
  - optional args:
    - names text doc: the text file listing the names of the image files to run (default 'image_file_names.txt')
    - circle only?: 
      - True: Only run circle detection 
      - False: Run circle detection and segementation (default)
    - threshold: Integer value for the threshold for seeded region growing algorithm (default 5)
  
#### Circle detection
  - True positive: correct circle found
  - False positive: wrong circle found
  - True negative: no implant present, no circle found (not applicable for our test set)
  - False positive: implant is present, but no circle is found

  ####Format for entering results in circle_detection.txt:
  #####version_model TP FP TN FN
  No spaces in version and model!
