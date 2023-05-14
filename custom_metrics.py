import pandas as pd 
import numpy as np 
import ast 

#from datasets import load_metric 
import sys 
import pickle
 
from difflib import SequenceMatcher
# metrics library 
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU
#from bert_score import BERTScorer
from rouge import Rouge 

cm = SmoothingFunction()

def GetSequenceMatch(a,b):
    """
    Inputs : a - sentence 1 
             b - sentence 2 
    Returns : sequence matching score of given sentence a and b 
    """
    return round(SequenceMatcher(None,a,b).ratio(),4)

def GetJaccardScore(a,b):
    """
   Inputs : a - sentence 1 
             b - sentence 2 
    Returns : Jaccard Similarity score of given sentence a and b 
    """
    w1 = set(a.lower().strip().split(' ')) 
    w2 = set(b.lower().strip().split(' '))
    
    intersection = w1.intersection(w2)
    union = w1.union(w2)
    # Calculate Jaccard similarity score 
    return round(float(len(intersection)) / len(union),4)

def get_self_bleu(hyps):
    sent_bleu_list = []
    for i in range(len(hyps)):
        h = hyps[i]
        r = hyps[:i] + hyps[i + 1:]
        sent_bleu_list.append(sentence_bleu(r, h, smoothing_function=cm.method2))
    return np.mean(sent_bleu_list)

def GetPredictionsSelfBleu(predictions):
    """
    Inputs : predictions - y_pred 
    """
    self_bleu = []
    for curr_pred in predictions:
        self_bleu.append(get_self_bleu(curr_pred))
    return np.round(np.mean(self_bleu),4)

def GetPredictionSequenceMatch(predictions,indicator,get_mean=False):
    """
    Inputs : predictions - y_pred (output generated by the model)
             indicator - can take two values 
                       -> sm - use sentence matching 
                       -> other(eg : js) - use jaccard similarity
    Returns : computes how similar the predicted headlines are. It returns three
              values (x,y,z), where x - similarity score between headline 1 and 
              headline 2, y - similarity score between headline 1 and headline 3,
              z - similarity score between headline 2 and headline 3. 
    """
    l12 = []
    l13 = []
    l23 = []
    if indicator == 'sm':
        #print("Sequence Matcher : ")
        for i in range(0,len(predictions)):
            sm12 = GetSequenceMatch(predictions[i][0],predictions[i][1])
            sm13 = GetSequenceMatch(predictions[i][0],predictions[i][2])
            sm23= GetSequenceMatch(predictions[i][1],predictions[i][2])
            l12.append(sm12)
            l13.append(sm13)
            l23.append(sm23)

    elif indicator == 'js':
        #print("Jaccard Similarity : ")
        for i in range(0,len(predictions)):
            sm12 = GetJaccardScore(predictions[i][0],predictions[i][1])
            sm13 = GetJaccardScore(predictions[i][0],predictions[i][2])
            sm23= GetJaccardScore(predictions[i][1],predictions[i][2])
            l12.append(sm12)
            l13.append(sm13)
            l23.append(sm23)

    elif indicator=='p_bleu':
        return GetPredictionsSelfBleu(predictions)
    
    elif indicator=='p_rouge':
        return GetPredictionsSelfRouge(predictions)
    
    if get_mean==True:
        return np.mean([np.round(np.mean(l12),4),np.round(np.mean(l13),4),np.round(np.mean(l23),4)])
    return np.round(np.mean(l12),4),np.round(np.mean(l13),4),np.round(np.mean(l23),4)

# use as json 
def Reference(path,pred=0):
    df= pd.read_csv(path,sep='\t',header=None)
    df.columns=['Output']
    y_true = []
    for curr in list(df['Output']):
        dicti = ast.literal_eval(curr)
        y_true.append(dicti['reference'])
    if pred==1:
        y_pred = []
        for curr in list(df['Output']):
            dicti = ast.literal_eval(curr)
            y_pred.append(dicti['predictions'][0])
        return y_pred 
    return y_true 

def GetReferences(path,pred=0):
    """
    Input : path - path to get all three headline references 
    Returns : y_true - reference headlines
    """
    y_true = []
    ref1 = Reference(path[0]) # B2 - H1 
    ref2 = Reference(path[1]) # B2 - H2
    ref3 = Reference(path[2]) # B2 - H3
    # Use context as dictionary 
    for r1,r2,r3 in zip(ref1,ref2,ref3):
        y_true.append([r1,r2,r3])
    if pred==1:
        y_pred = []
        pred1 = Reference(path[0],1)
        pred2 = Reference(path[1],1)
        pred3 = Reference(path[2],1)
        for p1,p2,p3 in zip(pred1,pred2,pred3):
            y_pred.append([p1,p2,p3])
        return y_pred 
    return y_true 

def GetMixtureContentPredictions(path):
    df= pd.read_csv(path,sep='\t',header=None)
    df.columns=['Output']
    y_pred = []
    for i in range(0,len(list(df['Output'])),3):
        h1 = df['Output'][i]
        h2 = df['Output'][i+1]
        h3 = df['Output'][i+2]
        y_pred.append([h1,h2,h3])
    return y_pred 

def MainModelPredictions(path):
    df= pd.read_csv(path,sep='\t',header=None)
    df.columns=['Output']
    y_pred = []
    my_dict = {}

    for i in range(len(list(df['Output']))):
        dicti = ast.literal_eval(df['Output'][i])
        id = int(dicti['instance_id'])
        if id not in my_dict:
            my_dict[id] = [dicti['predictions'][0]]
        else:
            my_dict[id].append(dicti['predictions'][0])
    
    for i in range(df.shape[0]//3):
        y_pred.append(my_dict[i+1])

    return y_pred 

def DatasetLoader(baseline_indicator,baseline1_path,baseline2_path,baseline3_path=None,main_model_path=None):
    """
    Input : baseline_indicator - can be 1,2,3,4 (having 4 types of baseline)
            baseline1_path, baseline2_path - output data available for two baselines
            baseline3_path, baseline4_path - not yet implemented 
    Returns : y_true,y_pred
    """
    y_pred = []
    y_true = GetReferences(baseline2_path)

    if baseline_indicator==1:
        df = pd.read_csv(baseline1_path,sep='\t',header = None)
        df.columns=['Output']
        for curr in list(df['Output']):
            dicti = ast.literal_eval(curr)
            y_pred.append(dicti['predictions'])
    elif baseline_indicator==2:
        y_pred = GetReferences(baseline2_path,1)
    elif baseline_indicator==3:
        y_pred = GetMixtureContentPredictions(baseline3_path)
    elif baseline_indicator=='M':
        y_pred = MainModelPredictions(main_model_path)
    else:
        print("Enter Valid Baseline number - Baseline "+main_model_path+" is invalid")
    
    return y_true, y_pred 

def MetricReferences(references,indicator='rouge'):
    """
    Inputs : references - y_true (reference headlines)
             indicator - can take three values 
                       -> rouge(default) - used for ROUGE Score references 
                       -> bleu - used for BLEU Score references
                       -> bert - used for BERT Score references
    Returns : list of reference headlines(for given metric) for given list of headlines 
    """
    new_ref = []
    if indicator=='bleu':
        #print("For BLEU")
        ref1 = []
        ref2 = []
        ref3 = []
        for i in range(len(references)):
            ref1.append(references[i][0])
            ref2.append(references[i][1])
            ref3.append(references[i][2])
        new_ref.extend([ref1,ref2,ref3])

    elif indicator=='bert':
        #print("For BERTScore")
        for i in range(len(references)):
            curr_ref = [references[i][0],references[i][1],references[i][2]]
            new_ref.extend([curr_ref])
            #print(new_ref)
    else:
        #print("ROUGE")
        for i in range(len(references)):
            curr_ref = references[i][0]+' '+references[i][1]+' '+references[i][2]
            new_ref.extend([curr_ref])
    #print(len(new_ref))
    return new_ref

def GetPredictedHeadlines(pred,indicator=1):
    """
    Inputs : pred - prediction data 
             indicator - 1(default) (headline1)
                       - 2(headline2)
                       - 3(headline3)
    """                 
    pred_res = []
    for i in range(len(pred)):
        pred_res.append(pred[i][indicator-1])
    #print(len(pred_res))
    return pred_res 

def AddResultsToFile(result,metric,filename):
    """
    Inputs : filename - name of result file 
             result - either ROUGE, BERTSCORE, BLEU Score 
             metric - print the current metric heading to the file 
    Gives the result file. 
    """
    try:
        headline_file = open(filename+'.txt', 'a')
        headline_file.write(metric+'\n\n')
        headline_file.write(str(result))
        headline_file.write('\n\n')
        headline_file.close()
    except:
        print("Unable to write to file")

def PairwiseRouge(predictions):
    rouge1 = []
    rouge2 = []
    rougel = []
    rouge = Rouge()
    for i in range(len(predictions)):
        h = predictions[i]
        r = predictions[:i] + predictions[i + 1:]
        curr_ref = ''
        for ref in r:
            curr_ref = curr_ref+' '+ref 

        score = rouge.get_scores(h,curr_ref,avg=True)
        rouge1.append(score['rouge-1']['f'])
        rouge2.append(score['rouge-2']['f'])
        rougel.append(score['rouge-l']['f'])
    return np.mean(rouge1), np.mean(rouge2), np.mean(rougel)

def GetPredictionsSelfRouge(predictions):
    """
    Inputs : predictions - y_pred 
    """
    self_rouge1 = []
    self_rouge2 = []
    self_rougel = []
    for curr_pred in predictions:
        r1,r2,rl = PairwiseRouge(curr_pred)
        self_rouge1.append(r1)
        self_rouge2.append(r2)
        self_rougel.append(rl)
    return np.round(np.mean(self_rougel),4)

def getHarmonicMean(a,b):
    return np.round(2*a*b/(a+b),4)

def HarmonicMeanResult(predictions,rougeL,bleu_score):
    self_rougeL = GetPredictionSequenceMatch(predictions,'p_rouge')
    self_bleu = GetPredictionsSelfBleu(predictions)
    hm_sb_bl = getHarmonicMean(1-self_bleu,bleu_score)
    hm_srL_rL = getHarmonicMean(1-self_rougeL,rougeL)
    return hm_sb_bl, hm_srL_rL

def GetResult(hypothesis,references,filename,p):
    """
    hypothesis - predicted headlines 
    references - reference headlines 
    """
    rouge_ref = MetricReferences(references,'rouge')
    #bert_ref = MetricReferences(references,'bert')
    bleu_ref = MetricReferences(references,'bleu')

    rouge = Rouge()
    ref1 = []
    ref2 = []
    ref3 = []
    for i in range(len(references)):
        ref1.append(references[i][0])
        ref2.append(references[i][1])
        ref3.append(references[i][2])

    rouge_scores1 = rouge.get_scores(hypothesis, ref1,avg=True)
    rouge_scores2 = rouge.get_scores(hypothesis, ref2,avg=True)
    rouge_scores3 = rouge.get_scores(hypothesis, ref3,avg=True)
    
    rouge_final = {}
    rouge_final['rouge-1_f1'] = round((rouge_scores1['rouge-1']['f']+rouge_scores2['rouge-1']['f']+rouge_scores3['rouge-1']['f'])/3,4)
    rouge_final['rouge-2_f1'] = round((rouge_scores1['rouge-2']['f']+rouge_scores2['rouge-2']['f']+rouge_scores3['rouge-2']['f'])/3,4)
    rouge_final['rouge-l_f1'] = round((rouge_scores1['rouge-l']['f']+rouge_scores2['rouge-l']['f']+rouge_scores3['rouge-l']['f'])/3,4)

    #rouge_scores = rouge.get_scores(hypothesis, rouge_ref,avg=True)
    #rouge_final = {}
    #rouge_final['rouge-1_f1'] = round(rouge_scores['rouge-1']['f'],4)
    #rouge_final['rouge-2_f1'] = round(rouge_scores['rouge-2']['f'],4)
    #rouge_final['rouge-l_f1'] = round(rouge_scores['rouge-l']['f'],4)

    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypothesis,bleu_ref)

    #bert = BERTScorer(lang="en", rescale_with_baseline=True)
    #bert_P, bert_R, bert_F1 = bert.score(hypothesis, bert_ref)
    #bert_final = {}
    #bert_final['F1'] =round(bert_F1.mean().item(),4)
    #bert_final['P'] = round(bert_P.mean().item(),4)
    #bert_final['R'] = round(bert_R.mean().item(),4)

    hm_sb_bl, hm_srL_rL = HarmonicMeanResult(p,rouge_final['rouge-l_f1'],bleu_score.score)
    AddResultsToFile(rouge_final,"ROUGE RESULT",filename)
    #AddResultsToFile(bert_final,"BERT RESULT",filename)
    AddResultsToFile(bleu_score,"BLEU RESULT",filename)
    AddResultsToFile(hm_sb_bl,"Harmonic Mean between Pairwise-Bleu score and Bleu score",filename)
    AddResultsToFile(hm_srL_rL,"Harmonic Mean between Pairwise-Rouge-L score and Rouge-L score",filename)

def AddHeadlineIndication(headline,filename):
    geeky_file = open(filename+'.txt', 'a')
    geeky_file.write('Headline '+str(headline)+' Result : '+'\n\n')
    geeky_file.close()

def AddBaselineIndication(baseline,filename):
    geeky_file = open(filename+'.txt', 'a')
    geeky_file.write('Baseline '+str(baseline)+' Result : '+'\n\n')
    geeky_file.close()

def BaselineResults(baseline,filename):
    AddBaselineIndication(baseline,filename)

    b1_path = "outputs/Baseline4-avgwmd-5050/Jun12_b4_topkp5095.tsv"
    b2_path1 = "outputs/jan30-baseline2-h1-exp-1/outputs-baseline2-h1-exp1.tsv"
    b2_path2 = "outputs/jan30-baseline2-h2-exp-1/outputs-baseline2-h2-exp1.tsv"
    b2_path3 = "outputs/jan30-baseline2-h3-exp-1/outputs-baseline2-h3-exp1.tsv"
    mix_path = "outputs/Baseline_Mixture_Selection/A4_MCS_Output_Pred.tsv"
    model_path = "outputs/DivNR00_check/J_prop_model_E10_1E4_5095.tsv"
    references, predictions = DatasetLoader(baseline,b1_path,[b2_path1,b2_path2,b2_path3],mix_path,model_path)

    AddResultsToFile(GetPredictionSequenceMatch(predictions,'sm'),'SEQUENCE MATCHER',filename)
    AddResultsToFile(GetPredictionSequenceMatch(predictions,'js'),'JACCARD SIMILARITY',filename)
    AddResultsToFile(GetPredictionSequenceMatch(predictions,'p_bleu'),'SELF-BLEU',filename)
    AddResultsToFile(GetPredictionSequenceMatch(predictions,'p_rouge'),'PAIRWISE-ROUGE',filename)

    AddHeadlineIndication(1,filename)
    GetResult(GetPredictedHeadlines(predictions,1),references,filename,predictions)
    AddHeadlineIndication(2,filename)
    GetResult(GetPredictedHeadlines(predictions,2),references,filename,predictions)
    AddHeadlineIndication(3,filename)
    GetResult(GetPredictedHeadlines(predictions,3),references,filename,predictions)

BaselineResults('M','outputs/DivNR00_check/Final_Result3_E10_1E4_5095')
#BaselineResults(2,'feb4-baseline1-beamsize-8-results')