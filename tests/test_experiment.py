import numpy
import pandas
from unittest import mock
from biomodelml.experiment import Experiment
from biomodelml.structs import DistanceStruct, TreeStruct, ImgDebug, ImgDebugs


def test_should_run_experiment(tmp_path):
    n1, n2 = "l1", "l2"
    v1 = mock.Mock()
    v1.name = n1
    d1 = DistanceStruct(
        ["f1", "f2", "f3", "f4"],
        numpy.array([
            numpy.array([0, 1, 0, 0]),
            numpy.array([1, 0, 0, 0]),
            numpy.array([0, 0, 0, 1]),
            numpy.array([0, 0, 1, 0])]))
    d2 = DistanceStruct(
        ["f1", "f2", "f3", "f4"],
        numpy.array([
            numpy.array([0, 0, 1, 0]),
            numpy.array([0, 0, 0, 1]),
            numpy.array([1, 0, 0, 0]),
            numpy.array([0, 1, 0, 0])]))
    v1.build_matrix = mock.Mock(return_value=d1)
    v2 = mock.Mock()
    v2.name = n2
    v2.build_matrix = mock.Mock(return_value=d2)
    exp = Experiment(tmp_path, v1, v2)
    assert type(exp.run()) == Experiment
    assert len(exp._trees) == 2
    assert type(exp._trees[0]) == TreeStruct
    assert type(exp._trees[1]) == TreeStruct
    assert exp._trees[0].name == n1
    assert exp._trees[0].distances == d1
    assert exp._trees[1].name == n2
    assert exp._trees[1].distances == d2
    v1.build_matrix.assert_called_once()
    v2.build_matrix.assert_called_once()

@mock.patch("biomodelml.experiment.pandas")
def test_should_save_experiment(mock_pandas, tmp_path):
    mock_pandas.DataFrame = mock.Mock(return_value=pandas.DataFrame([1,2,3]))
    v1 = mock.Mock()
    v2 = mock.Mock()
    exp = Experiment(tmp_path, v1, v2)
    t1 = mock.Mock()
    t1.name = "t1"
    t1.distances.align.get_gapped_sequences = mock.Mock(return_value=["a", "b", "c"])
    t1.distances.img_debugs = []
    t1.distances.names = ["C", "A", "B"]
    t1.tree.to_newick = mock.Mock(return_value="(1,2);")
    t2 = mock.Mock()
    t2.name = "t2"
    t2.distances.align.get_gapped_sequences = mock.Mock(return_value=["a", "b", "c"])
    t2.distances.img_debugs = []
    t2.distances.names = ["C", "A", "B"]
    t2.tree.to_newick = mock.Mock(return_value="(1,0);")
    exp._trees = [t1, t2]
    exp.save()
    assert (tmp_path / "t1.png").exists()
    assert (tmp_path / "t2.png").exists()
    assert (tmp_path / "t1.fasta").exists()
    assert (tmp_path / "t2.fasta").exists()
    assert (tmp_path / "t1.csv").exists()
    assert (tmp_path / "t2.csv").exists()
    assert (tmp_path / "t1.nw").exists()
    assert (tmp_path / "t2.nw").exists()
    assert (tmp_path / "t1.map").exists() == False
    assert (tmp_path / "t2.map").exists() == False


@mock.patch("biomodelml.experiment.pandas")
def test_should_save_experiment_without_alignment(mock_pandas, tmp_path):
    mock_pandas.DataFrame = mock.Mock(return_value=pandas.DataFrame([1,2,3]))
    v1 = mock.Mock()
    v2 = mock.Mock()
    exp = Experiment(tmp_path, v1, v2)
    t1 = mock.Mock()
    t1.name = "t1"
    t1.distances.align = False
    t1.distances.names = ["C", "A", "B"]
    t1.distances.img_debugs = []
    t1.tree.to_newick = mock.Mock(return_value="(1,2);")
    t2 = mock.Mock()
    t2.name = "t2"
    t2.distances.align = False
    t2.distances.names = ["C", "A", "B"]
    t2.distances.img_debugs = []
    t2.tree.to_newick = mock.Mock(return_value="(1,0);")
    exp._trees = [t1, t2]
    exp.save()
    assert (tmp_path / "t1.png").exists()
    assert (tmp_path / "t2.png").exists()
    assert (tmp_path / "t1.fasta").exists() == False
    assert (tmp_path / "t2.fasta").exists() == False
    assert (tmp_path / "t1.csv").exists()
    assert (tmp_path / "t2.csv").exists()
    assert (tmp_path / "t1.nw").exists()
    assert (tmp_path / "t2.nw").exists()
    assert (tmp_path / "t1.map").exists() == False
    assert (tmp_path / "t2.map").exists() == False


@mock.patch("biomodelml.experiment.pandas")
def test_should_save_experiment_with_img_debug(mock_pandas, tmp_path):
    mock_pandas.DataFrame = mock.Mock(return_value=pandas.DataFrame([1,2,3]))
    v1 = mock.Mock()
    v2 = mock.Mock()
    exp = Experiment(tmp_path, v1, v2)
    t1 = mock.Mock()
    t1.name = "t1"
    t1.distances.align = False
    t1.distances.names = ["C", "A", "B"]
    t1.distances.img_debugs = [ImgDebugs("img1", "img2", [ImgDebug("1.0", "2", "3", "2", "3", "10")])]
    t1.tree.to_newick = mock.Mock(return_value="(1,2);")
    t2 = mock.Mock()
    t2.name = "t2"
    t2.distances.align = False
    t2.distances.names = ["C", "A", "B"]
    t2.distances.img_debugs = [ImgDebugs("img1", "img2", [ImgDebug("1.0", "2", "3", "2", "3", "20")])]
    t2.tree.to_newick = mock.Mock(return_value="(1,0);")
    exp._trees = [t1, t2]
    exp.save()
    assert (tmp_path / "t1.map").exists()
    assert (tmp_path / "t2.map").exists()
