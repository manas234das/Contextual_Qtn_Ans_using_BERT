####################################################################################################################################################

## Function to get the small paragraphts for BERT to work
def subPara(paragraph):
    # paragraph length
    para_len = len(paragraph)
    _final_para = []
    if(para_len > 18):
        _final_para.append(".".join(paragraph[0:para_len//2]))
        _final_para.append(".".join(paragraph[para_len//2:para_len+1]))
        return _final_para
    else:
        return '.'.join(paragraph)

def samplePara(paragraph_list):
    """
    For a longer paragraph, this function splits the paragraph into two small ones which the BERT model can take
    
    Arguments:
        paragraph_list {list} : List of all the paragraphs
    """
    
    # takenize the para and send
    
    sub_paragraphs = []
    for i, par in enumerate(paragraph_list):
        _para = par.split(".")
        sub_paragraphs.append(subPara(_para))
    ## Extracting the list of sentences
    final_para = []
    for i, para in enumerate(sub_paragraphs):
        if(type(para) == list):
            for sub_par in para:
                final_para.append(sub_par)
        else:
            final_para.append(para)
    return final_para
            
####################################################################################################################################################




