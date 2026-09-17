import copy
import json
import numpy as np
import pytest
import torch
from doppler_jepa.geometry import Grid, world_to_reference, points_to_grid, rasterize_polygons
from doppler_jepa.model import Forecaster, forward_splat
from doppler_jepa.losses import latent_loss, feature_regularizer, forecast_loss, reconstruction_loss
from doppler_jepa.metrics import BinaryMetrics
from doppler_jepa.prepare import scene_partition, make_windows
from doppler_jepa.synthetic import generate
from doppler_jepa.data import RadarWindows


torch.set_num_threads(2)


def test_rigid_transform():
    theta=.7
    R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    t=np.array([100.,200.,0.])
    local=np.array([[3.,4.,0.]])
    world=local@R.T+t
    assert np.allclose(world_to_reference(world,R,t),local)


def test_polygon_and_point_grid():
    grid=Grid(size=32,xmax=32,ymin=-16,ymax=16)
    polygon=np.array([[[5,1,0],[5,-1,0],[9,-1,0],[9,1,0]]])
    mask=rasterize_polygons(polygon,grid)
    assert mask.sum()==8
    p=np.array([[6.,0.,0.,1.,0.,0.,3.,10.,-.1]])
    raster=points_to_grid(p,np.eye(3),np.zeros(3),grid)
    assert raster[5].sum()==1
    assert np.isclose(raster[2].max(),.15)
    assert points_to_grid(p,np.eye(3),np.zeros(3),grid,True)[2:4].sum()==0


def test_forward_splat_direction_and_gradient():
    feature=torch.zeros(1,2,5,5,requires_grad=True)
    with torch.no_grad():feature[0,:,1,2]=torch.tensor([2.,3.])
    velocity=torch.zeros(1,2,5,5);velocity[:,0]=1
    support=torch.zeros(1,1,5,5);support[0,0,1,2]=1
    result,mass=forward_splat(feature,velocity,support,torch.ones(1),1,1)
    assert torch.allclose(result[0,:,2,2],torch.tensor([2.,3.]))
    assert mass[0,0,2,2]==1
    result.sum().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


@pytest.mark.parametrize('transport',[False,True])
def test_model_loss_and_backward(transport):
    model=Forecaster(16,transport,2.,2.)
    x=torch.rand(2,4,6,32,32);x[:,:,5]=(x[:,:,5]>.9).float()
    dt=torch.tensor([[.5,1,1.5]]).repeat(2,1)
    out=model(x,dt)
    assert out['logits'].shape==(2,3,32,32)
    loss=forecast_loss(out['logits'],torch.rand(2,3,32,32)>.95,torch.ones(2,32,32))
    loss.backward()
    assert torch.isfinite(loss)


def test_teacher_stop_gradient_and_mask():
    model=Forecaster(16,True,2.,2.)
    teacher=copy.deepcopy(model.encoder).requires_grad_(False)
    x=torch.rand(2,4,6,32,32);x[:,:,5]=(x[:,:,5]>.8).float()
    future=torch.rand(2,3,6,32,32);future[:,:,5]=(future[:,:,5]>.8).float()
    out=model(x,torch.ones(2,3),task='jepa')
    with torch.no_grad():z=teacher(future.flatten(0,1)).reshape(2,3,16,8,8)
    variance,covariance,_=feature_regularizer(out['context'],x)
    loss=latent_loss(out['latent'],z,future,torch.ones(2,32,32))+variance+.01*covariance
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters())
    empty=future.clone();empty[:,:,5]=0
    assert latent_loss(out['latent'],z,empty,torch.ones(2,32,32)).item()==0


def test_reconstruction_baseline():
    m=Forecaster(16,False,2,2)
    x=torch.rand(2,4,6,32,32);target=torch.rand(2,3,6,32,32)
    target[:,:,5]=(target[:,:,5]>.8).float()
    out=m(x,torch.ones(2,3),task='reconstruction')
    loss=reconstruction_loss(out['reconstruction'],target,torch.ones(2,32,32))
    loss.backward();assert torch.isfinite(loss)


def test_known_metrics():
    m=BinaryMetrics(.5)
    m.update(np.array([.9,.8,.2,.1]),np.array([1,0,1,0]))
    r=m.result()
    assert r['tp']==1 and r['fp']==1 and r['fn']==1 and r['tn']==1
    assert np.isclose(r['iou'],1/3)
    assert np.isclose(r['ap_hist4096'],(1+2/3)/2)
    assert np.isclose(r['brier'],(.01+.64+.64+.01)/4)


def test_scene_partition():
    splits=scene_partition([f's{i}' for i in range(20)],['v1','v2'])
    assert not set(splits['train'])&set(splits['dev'])
    assert len(splits['dev'])==2


def test_future_target_overlap_rejected():
    records=[{'token':str(i),'scene':'s','timestamp':i*500000,'single_min_us':i*500000-1000} for i in range(10)]
    assert len(make_windows(records,4,3))==4
    for r in records:r['single_min_us']=0
    assert not make_windows(records,4,3)


def test_missing_causal_placeholder_breaks_windows():
    records=[{'token':str(i),'scene':'s','timestamp':i*500000,
              'single_min_us':i*500000-1000,'valid':True} for i in range(10)]
    assert len(make_windows(records,4,3))==4
    records[0]['valid']=False
    # Only the first candidate window touches frame 0; no temporal bridging occurs.
    assert len(make_windows(records,4,3))==3


@pytest.fixture(scope='module')
def cache(tmp_path_factory):
    return generate(tmp_path_factory.mktemp('radar_cache'))


def test_dataset_modalities_and_shapes(cache):
    data=RadarWindows(cache,'train',grid=Grid(size=32))
    sample=data[0]
    assert sample['x'].shape==(4,6,32,32)
    assert sample['y'].shape==(3,32,32)
    assert sample['y'].sum()>0
    ssl=RadarWindows(cache,'train',purpose='ssl',grid=Grid(size=32))[0]
    assert 'y' not in ssl and 'moving' not in ssl
    assert ssl['future'].shape==(3,6,32,32)
    assert set(ssl)=={'x','dt','valid','token','scene','future'}


def test_deterministic_corruption_and_clean_targets(cache):
    clean=RadarWindows(cache,'train',grid=Grid(size=32))[0]
    a=RadarWindows(cache,'train',grid=Grid(size=32),corruption='dropout',severity=.5)[0]
    b=RadarWindows(cache,'train',grid=Grid(size=32),corruption='dropout',severity=.5)[0]
    assert torch.equal(a['x'],b['x']) and torch.equal(a['y'],clean['y'])
    assert (a['x'][:,5].sum()<=clean['x'][:,5].sum())


def test_label_fractions_nested(cache):
    a=RadarWindows(cache,'train',fraction=.25)
    b=RadarWindows(cache,'train',fraction=.5)
    assert set(a.scenes)<=set(b.scenes)
    assert len(a.scenes)==1 and len(b.scenes)==2


def test_zero_input_finite(cache):
    sample=RadarWindows(cache,'train',grid=Grid(size=32),corruption='burst',severity=4)[0]
    assert sample['x'].sum()==0
    model=Forecaster(16,True,2,2)
    out=model(sample['x'][None],sample['dt'][None])
    assert torch.isfinite(out['logits']).all()


def test_age_aware_prior_falls_back_during_outage():
    from doppler_jepa.model import recent_transport_prior
    x=torch.zeros(1,4,6,8,8)
    x[0,1,5,2,2]=1
    x[0,1,2,2,2]=.1  # 2 m/s after denormalization.
    x[0,1,4,2,2]=-1.0
    velocity,support,age=recent_transport_prior(x,(8,8))
    assert velocity[0,0,2,2]==2
    assert support[0,0,2,2]==1
    assert age[0,0,2,2]==-1
    feature=torch.ones(1,1,8,8)
    result,mass=forward_splat(feature,velocity,support,torch.ones(1,8,8)-age[:,0],1,1)
    assert mass[0,0,6,2]==1  # At t+1, observation from t-1 moves 2 seconds * 2m/s.


def test_age_ablation_changes_only_fixed_prior():
    torch.manual_seed(4)
    a = Forecaster(width=16, age_aware=True).eval()
    b = Forecaster(width=16, age_aware=False).eval()
    b.load_state_dict(a.state_dict())
    x = torch.zeros(1, 2, 6, 32, 32)
    x[:, 0, 5, 8:20, 8:20] = 1
    x[:, 0, 2, 8:20, 8:20] = .1
    x[:, 0, 4, 8:20, 8:20] = -1
    dt = torch.ones(1, 1)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    with torch.no_grad():
        first = a(x, dt)['logits']
        second = b(x, dt)['logits']
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.allclose(first, second)
