"""LivePortrait head and original texture with a DH_live Mini mouth."""
import gzip
import json
import os
import sys
import time

from avatar_paths import DH as REPO
from avatar_geometry import Identity, read, face_tracker, face_points, mp, torch, cv2, np
from avatar_layers import subject_layers

sys.path.insert(0, str(REPO))

LIP_SUPPORT_RADIUS=0
UPPER_TEETH_FIT=.045


def render_offscreen(renderer,vertBuffer,out_size=(368,368),mat_world=None,bs_array=None):
    """Read the rendered back buffer directly; never present a hidden window.

    DH's original helper swaps the hidden window twice per mouth frame. That
    unnecessarily brings desktop presentation into this offscreen pipeline.
    """
    import glfw,glm
    from OpenGL.GL import (glUseProgram,glUniformMatrix4fv,glGetUniformLocation,GL_FALSE,
        glEnable,GL_DEPTH_TEST,GL_BLEND,glBlendFunc,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,
        GL_CULL_FACE,glCullFace,GL_BACK,glFrontFace,GL_CW,glClearColor,glClear,
        GL_COLOR_BUFFER_BIT,GL_DEPTH_BUFFER_BIT,glUniform1i,glUniform1fv,glUniform2fv,
        glBindVertexArray,glDrawElements,GL_TRIANGLES,GL_UNSIGNED_INT,glDrawBuffer,
        glReadBuffer,glReadPixels,GL_RGBA,GL_UNSIGNED_BYTE)
    glfw.make_context_current(renderer.window)
    glUseProgram(renderer.program);glDrawBuffer(GL_BACK)
    projection=glm.ortho(0,out_size[0],0,out_size[1],1000,-1000)
    glUniformMatrix4fv(glGetUniformLocation(renderer.program,'gProjection'),1,GL_FALSE,glm.value_ptr(projection))
    glEnable(GL_DEPTH_TEST);glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_CULL_FACE);glCullFace(GL_BACK);glFrontFace(GL_CW)
    glClearColor(.5,.5,.5,0);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glUniform1i(glGetUniformLocation(renderer.program,'texture_bs'),0)
    glUniformMatrix4fv(glGetUniformLocation(renderer.program,'gWorld0'),1,GL_FALSE,mat_world)
    glUniform1fv(glGetUniformLocation(renderer.program,'bsVec'),12,bs_array.astype(np.float32))
    glUniform2fv(glGetUniformLocation(renderer.program,'vertBuffer'),209,vertBuffer.astype(np.float32))
    glBindVertexArray(renderer.vao);glDrawElements(GL_TRIANGLES,renderer.indices.size,GL_UNSIGNED_INT,None);glBindVertexArray(0)
    glReadBuffer(GL_BACK)
    data=glReadPixels(0,0,*renderer.window_size,GL_RGBA,GL_UNSIGNED_BYTE,outputType=None)
    return data.reshape(renderer.window_size[1],renderer.window_size[0],-1).astype(np.uint8)


class Head:
    def __init__(self, profile):
        self.portrait_path = profile / 'portrait.png'
        self.source = read(self.portrait_path)
        if self.source is None or self.source.shape != (720, 720, 3):
            raise ValueError('Prepare the profile with scripts/prepare_portrait.py first.')
        with face_tracker(mp.tasks.vision.RunningMode.IMAGE) as detector:
            points = face_points(detector, self.source)
        if points is None:
            raise ValueError('No face in the prepared portrait.')
        self.model = Identity(self.source, points)
        self.gray = cv2.resize(self.model.gray, (256, 256), interpolation=cv2.INTER_AREA)
        with gzip.open(profile / 'combined_data.json.gz', 'rt', encoding='utf8') as file:
            self.combined = json.load(file)
        row=self.combined['json_data'][0]
        self.rect=np.array(row['rect'],np.int32)
        self.base_points=np.array(row['points'][16:],np.float32).reshape(-1,2)
        self.base_matrix=np.array(row['points'][:16],np.float32).reshape(4,4).T
        full=self.base_points*(self.rect[2]-self.rect[0])/184+self.rect[:2]
        self.base_crop=cv2.transform(full.astype(np.float32)[None],self.model.affine*2)[0]
        self.affine_fit=np.linalg.pinv(np.column_stack([self.base_points,np.ones(len(self.base_points))]))
        self.xy=np.stack(np.meshgrid(np.arange(720),np.arange(720)),axis=-1).astype(np.float32)
        self.mask=cv2.warpAffine(self.model.mask,self.model.inverse,(720,720))
        self.background=self.source.astype(np.float32)*(1-self.mask[...,None])
        yy,xx=np.nonzero(self.mask>0)
        self.roi=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
        self.roi_inverse=self.model.inverse.copy();self.roi_inverse[:,2]-=self.roi[:2]
        background = profile / 'background.png'
        self.body_plate, self.body_alpha = subject_layers(
            self.source, background if background.is_file() else None,
        )
        self.body_plate = self.body_plate.astype(np.float32)
        self.body_alpha = np.repeat(self.body_alpha[..., None], 3, axis=2)

    @torch.inference_mode()
    def render(self,pose):
        start=time.perf_counter();model=self.model
        raw=model.render(*pose[:4])
        gray=cv2.resize(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY),(256,256),interpolation=cv2.INTER_AREA)
        flow=cv2.calcOpticalFlowFarneback(gray,self.gray,None,.5,3,15,3,5,1.2,0)
        flow=cv2.resize(flow,(512,512))*2
        coords=model.grid+flow
        reference=cv2.remap(model.neutral,coords[...,0],coords[...,1],cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        eye=cv2.remap(model.eye_mask,coords[...,0],coords[...,1],cv2.INTER_LINEAR)
        weight=1.-eye*pose[3]
        # Sample the original photograph once at output resolution. The old path
        # enlarged it to 512, warped it there, then resampled the texture again.
        x0,y0,x1,y1=self.roi;size=(x1-x0,y1-y0)
        source_coords=cv2.transform(coords,model.inverse)
        source_coords=cv2.warpAffine(source_coords,self.roi_inverse,size)
        original=cv2.remap(self.source,source_coords[...,0],source_coords[...,1],cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        texture_alpha=cv2.warpAffine(weight*model.mask,self.roi_inverse,size)
        residual=raw.astype(np.float32)-reference*weight[...,None]
        layer=cv2.warpAffine(residual*model.mask[...,None],self.roi_inverse,size)+original*texture_alpha[...,None]
        frame=self.source.copy();frame[y0:y1,x0:x1]=np.clip(layer+self.background[y0:y1,x0:x1],0,255).astype(np.uint8)
        # Invert the backward texture flow at the reference DH vertices. This
        # avoids a second, noisy face detector and full-sequence preparation.
        q=self.base_crop.copy()
        for _ in range(3):
            displacement=cv2.remap(flow,q[:,0,None],q[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,0]
            q=self.base_crop-displacement
        full=cv2.transform(q[None],model.inverse)[0]
        points=(full-self.rect[:2])*(184/(self.rect[2]-self.rect[0]))
        affine=(self.affine_fit@points).T
        transform=np.eye(4,dtype=np.float32)
        transform[0,[0,1,3]]=affine[0];transform[1,[0,1,3]]=affine[1]
        matrix=transform@self.base_matrix
        return {'frame':frame,'points':points,'matrix':matrix,'rect':self.rect,
                'body':pose[4],'head_ms':(time.perf_counter()-start)*1000}

    def compose_body(self,frame,body):
        # One foreground layer, pivot at the seated torso. Background stays fixed.
        roll,tx,ty,scale,breath=body
        affine=cv2.getRotationMatrix2D((360,700),float(roll),float(scale)).astype(np.float32);affine[:,2]+=[tx,ty]
        alpha=cv2.warpAffine(self.body_alpha,affine,(720,720),borderMode=cv2.BORDER_REPLICATE)
        subject=cv2.warpAffine(frame,affine,(720,720),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        foreground=cv2.multiply(subject,alpha,dtype=cv2.CV_32F)
        background=cv2.multiply(self.body_plate,cv2.subtract((1.,1.,1.,1.),alpha))
        return cv2.convertScaleAbs(cv2.add(foreground,background))

    def midpoint(self,a,b):
        ga=cv2.resize(cv2.cvtColor(a['frame'],cv2.COLOR_BGR2GRAY),(256,256))
        gb=cv2.resize(cv2.cvtColor(b['frame'],cv2.COLOR_BGR2GRAY),(256,256))
        flow=cv2.calcOpticalFlowFarneback(ga,gb,None,.5,3,15,3,5,1.2,0)
        flow=cv2.resize(flow,(720,720))*(720/256)*.5
        # Only the head ROI differs in the two images. Keep the photographic
        # surroundings intact and avoid two cubic warps of the whole frame.
        pad=int(np.ceil(np.abs(flow).max()))+4
        x0,y0,x1,y1=self.roi
        x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(720,x1+pad);y1=min(720,y1+pad)
        flow=flow[y0:y1,x0:x1];grid=self.xy[y0:y1,x0:x1]
        ca=grid-flow;cb=grid+flow
        ia=cv2.remap(a['frame'],ca[...,0],ca[...,1],cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        ib=cv2.remap(b['frame'],cb[...,0],cb[...,1],cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        body=(a['body']+b['body'])/2
        frame=a['frame'].copy();frame[y0:y1,x0:x1]=cv2.addWeighted(ia,.5,ib,.5,0)
        return {'frame':frame,'points':(a['points']+b['points'])/2,
                'matrix':(a['matrix']+b['matrix'])/2,'body':body,'rect':self.rect,'head_ms':0.}


class Mouth:
    def __init__(self,combined,upper_teeth_fit=UPPER_TEETH_FIT):
        self.articulation_gain=1.30
        self.lip_support_radius=LIP_SUPPORT_RADIUS
        os.chdir(REPO)
        # Native OpenCV builds need byte-buffer loading for Cyrillic paths.
        cv2.imread=lambda path,flags=cv2.IMREAD_COLOR:cv2.imdecode(np.fromfile(path,np.uint8),flags)
        from talkingface.render_model_mini import RenderModel_Mini
        from mini_live.render import create_render_model
        from OpenGL.GL import GL_VERTEX_SHADER,GL_FRAGMENT_SHADER
        from OpenGL.GL.shaders import compileProgram,compileShader
        self.model=RenderModel_Mini();self.model.loadModel(str(REPO/'checkpoint/DINet_mini/epoch_40_new.pth'))
        self.model.net.infer_model.ref_in_feature=torch.tensor(combined['ref_data'][:64],dtype=torch.float32,device='cuda').reshape(1,-1)
        self.renderer=create_render_model((368,368),floor=20)
        from types import MethodType
        self.renderer.render2cv=MethodType(render_offscreen,self.renderer)
        vertex=(REPO/'mini_live/shader/prompt3.vsh').read_text()
        fragment='''#version 330
in vec2 v_texture; in vec2 v_bias; out vec4 out_color;
void main(){if(v_texture.x>=3.0 || v_texture.x==-2.0)discard;out_color=vec4((v_bias+1.0)/2.0,0.5,1.0);}'''
        self.texture_program=compileProgram(compileShader(vertex,GL_VERTEX_SHADER),compileShader(fragment,GL_FRAGMENT_SHADER))
        self.vertices=np.array([[float(x) for x in line[2:].split()] for line in combined['face3D_obj'] if line.startswith('v ')]).reshape(-1,5)
        from mini_live.obj.wrap_utils import index_wrap
        # Upstream personalizes the lip vertices but retains a generic tooth
        # mesh. Its lower edge sat at this person's upper lip, hiding the teeth
        # even on open vowels. Fit its vertical placement to the personal mouth.
        mouth_width=np.linalg.norm(self.vertices[index_wrap.index(78),:3]-self.vertices[index_wrap.index(308),:3])
        self.vertices[self.vertices[:,3]==3,1]+=mouth_width*upper_teeth_fit
        self.renderer.GenVBO(self.vertices)
        self.blend=cv2.cvtColor(cv2.imread(str(REPO/'mini_live/bs_texture_halfFace.png')),cv2.COLOR_BGR2RGB)[:6,:209].astype(np.float32)/255*2-1
        inner=[78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
        self.inner=[index_wrap.index(i) for i in inner]
        self.fusion_mask=self.model.net.face_fusion_tensor[0,0].cpu().numpy()[...,None]
        self.native_grids={}

    @torch.inference_mode()
    def render(self,head,coeff,speech_weight=1.):
        if speech_weight<=0:return head['frame'].copy()
        from OpenGL.GL import glUseProgram
        from talkingface.data.few_shot_dataset import get_image
        import torch.nn.functional as F
        frame=cv2.cvtColor(head['frame'],cv2.COLOR_BGR2RGBA)
        matrix=head['matrix'].T*2;points=head['points'];rect=head['rect']
        bs=np.zeros(12,np.float32);bs[:6]=np.array(coeff[:6],np.float32)*self.articulation_gain
        glUseProgram(self.renderer.program)
        prompt=self.renderer.render2cv(points/184*2-1,out_size=(368,368),mat_world=matrix,bs_array=bs)[::2,::2]
        source=get_image(frame,rect,input_type='image',resize=184)
        st=torch.from_numpy(source.astype(np.float32)/255).permute(2,0,1)[None].cuda()
        pt=torch.from_numpy(prompt.astype(np.float32)/255).permute(2,0,1)[None].cuda()
        generated=self.model.interface(st,pt)
        program=self.renderer.program;self.renderer.program=self.texture_program
        try:
            glUseProgram(self.texture_program)
            tp=self.renderer.render2cv(points/184*2-1,out_size=(368,368),mat_world=matrix,bs_array=bs)[::2,::2]
        finally:self.renderer.program=program;glUseProgram(program)
        v=self.vertices[:209,:3]+(self.blend*bs[:6,None,None]).sum(0)
        target=(head['matrix']@np.column_stack([v,np.ones(len(v))]).T).T[:,:2]
        mask=np.zeros((184,184),np.float32)
        cv2.fillPoly(mask,[np.rint(target[self.inner]*8).astype(np.int32)],1.,shift=3)
        # Synthesize only the cavity. Extending the mask over the lip surface
        # projects teeth/bright texture over the lip edge, especially on closure.
        r=self.lip_support_radius
        if r:mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
        mask=cv2.GaussianBlur(mask,(0,0),.65)
        x0,y0,x1,y1=rect;w=x1-x0;h=y1-y0
        size=(w,h)
        if size not in self.native_grids:self.native_grids[size]=np.stack(np.meshgrid(np.arange(w),np.arange(h)),axis=-1).astype(np.float32)
        displacement=(tp[...,:2].astype(np.float32)/255*2-1)*self.fusion_mask
        displacement=cv2.resize(displacement,size,interpolation=cv2.INTER_LINEAR)*np.array([w/2,h/2],np.float32)
        sample=self.native_grids[size]-displacement
        original=frame[y0:y1,x0:x1,:3]
        real=cv2.remap(original,sample[...,0],sample[...,1],cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        cavity=generated[0,:3].permute(1,2,0).clamp(0,1).mul(255).byte().cpu().numpy()
        cavity=cv2.resize(cavity,size,interpolation=cv2.INTER_CUBIC)
        alpha=cv2.resize(mask,size,interpolation=cv2.INTER_LINEAR)[...,None]
        patch=cavity*alpha+real*(1-alpha)
        frame[y0:y1,x0:x1,:3]=np.clip(patch*speech_weight+original*(1-speech_weight),0,255).astype(np.uint8)
        return cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR)

    def close(self):
        import glfw
        glfw.destroy_window(self.renderer.window);glfw.terminate()
