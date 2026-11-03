from compare_m3_eight import usage_summary, score_metrics


def test_unknown_memory_usage_is_not_claimed_complete():
    value=usage_summary([{'status':'returned','usage':{'prompt_tokens':100,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':40}}},
                         {'status':'error','usage':None}])
    assert value['total_tokens']==120 and value['unknown_usage_calls']==1
    assert not value['complete_usage'] and value['uncached_prompt_tokens'] is None


def test_qa_cached_and_completion_counts():
    value=usage_summary([{'status':'returned','usage':{'prompt_tokens':100,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':40}}}])
    assert value['uncached_prompt_tokens']==60 and value['total_tokens']==120 and value['complete_usage']


def test_all_questions_in_denominator():
    rows=[{'judge_error':False,'judge_score':0,'error':'timeout','predicted_answer':''},
          {'judge_error':False,'judge_score':1,'error':'','predicted_answer':'yes'}]
    m=score_metrics(rows);assert m['mean_score']==.5 and m['full_credit_fraction']==.5 and m['questions']==2
    rows[1]['judge_error']=True
    assert score_metrics(rows)['mean_score'] is None
