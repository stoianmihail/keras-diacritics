# Role:
# It takes an already trained model (ann) and solve the diacritics from infile 

def serve(ann, infile):
    outfile = open(infile.replace(".txt", "") + ".ok", "w")
    
    with open(infile, 'r') as myfile:
        data = myfile.read()
        
    outfile.write(ann.predict(data))
    pass
