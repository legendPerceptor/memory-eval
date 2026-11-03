import pytest
from retry_m3_ten_timeouts import select_timeouts


def sample():
    q=dict(video_id='kitchen_01', question_id='kitchen_01_Q13', question_index=12,
           question='test', answer='expected', timestamp='36:21', before_clip=71)
    old={**q, 'expected_answer':'expected', 'finish_reason':'timeout', 'error':'timeout'}
    return q,old


def test_original_indices_survive_filtering():
    q,r=sample(); result=select_timeouts([q],[r],[q['question_id']])
    assert result[0]['question_index']==12
    assert result[0] is q


@pytest.mark.parametrize('field,value',[('finish_reason','done'),('error',''),('question_index',0),('expected_answer','changed')])
def test_only_unchanged_original_timeouts_can_be_retried(field,value):
    q,r=sample();r[field]=value
    with pytest.raises(ValueError):select_timeouts([q],[r],[q['question_id']])


def test_duplicates_are_rejected():
    q,r=sample()
    with pytest.raises(ValueError):select_timeouts([q],[r],[q['question_id']]*2)
    with pytest.raises(ValueError):select_timeouts([q,q],[r],[q['question_id']])
