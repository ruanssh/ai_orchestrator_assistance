from sqms_ai_orchestrator.orchestrator import AIOrchestrator


def test_organizational_questions_are_detected():
    assert AIOrchestrator._is_organizational('Quem é o gerente de TI?')
    assert AIOrchestrator._is_organizational('Quem é responsável por Procurement?')
    assert AIOrchestrator._is_organizational('Qual o cargo de Nadyson Oliveira?')


def test_operational_question_is_not_organizational():
    assert not AIOrchestrator._is_organizational('Como criar uma cotação no SQMS?')
