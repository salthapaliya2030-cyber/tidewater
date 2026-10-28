import { GPU } from './gpu/GPU.js';
import { PerspectiveCamera } from './scene/Camera.js';
import { Scene } from './scene/Scene.js';
import { Timer } from './math/Timer.js';
import { MeshRenderer } from './render/MeshRenderer.js';
import { FrameUniforms } from './render/Frame.js';

// Canvas, device, main camera / scene and the frame loop.
export class Engine {

	constructor( container ) {

		this.container = container;
		this.renderScale = 1;
		this.clock = new Timer();
		this.frame = 0;
		this.onResize = [];

	}

	async init() {

		const canvas = document.createElement( 'canvas' );
		canvas.tabIndex = 0;
		this.container.appendChild( canvas );
		this.canvas = canvas;
		this.domElement = canvas;
		// Let the GPU layer surface validation/device-loss errors to users without DevTools.
		window.__showTidewaterError = ( title, detail ) => this.showRuntimeError( title, detail );
		await GPU.init( { canvas } );
		this.meshRenderer = new MeshRenderer();
		this.meshRenderer.syncPipelines = false; // compile in the background (App.precompile waits for them)
		this.camera = new PerspectiveCamera( 62, window.innerWidth / window.innerHeight, 0.06, 60000 );
		this.scene = new Scene();
		window.addEventListener( 'resize', () => this.resize() );
		this.resize();

	}

	setRenderScale( s ) {

		this.renderScale = s;
		this.resize();

	}

	// output (canvas) size in pixels
	get width() {

		return this.canvas.width;

	}

	get height() {

		return this.canvas.height;

	}

	resize() {

		const w = window.innerWidth, h = window.innerHeight;
		const dpr = this.renderScale;
		this.canvas.width = Math.max( 1, Math.floor( w * dpr ) );
		this.canvas.height = Math.max( 1, Math.floor( h * dpr ) );
		this.canvas.style.width = w + 'px';
		this.canvas.style.height = h + 'px';
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
		FrameUniforms.fields.outputResolution.value.set( this.canvas.width, this.canvas.height );
		for ( const f of this.onResize ) f( w, h );

	}

	// the canvas texture of this frame (render target of the final post pass)
	currentTexture() {

		return GPU.context.getCurrentTexture();

	}

	start( update ) {

		const loop = ( t ) => {

			this.clock.update( t );
			let dt = this.clock.getDelta();
			if ( dt > 0.1 ) dt = 0.1;
			this.frame ++;
			try {

				update( dt, this.clock.getElapsed() );

			} catch ( error ) {

				// A frame error used to stop requestAnimationFrame silently, leaving the UI over a black canvas.
				// Show the actual runtime / WebGPU pipeline error on screen so it can be diagnosed without DevTools.
				console.error( 'Tidewater frame stopped:', error );
				this.stop();
				this.showRuntimeError( 'Tidewater stopped rendering', error );
				return;

			}
			this._raf = requestAnimationFrame( loop );

		};

		this._raf = requestAnimationFrame( loop );

	}

	showRuntimeError( title, detail ) {

		const message = String( detail && ( detail.stack || detail.message ) || detail || 'No additional details were provided.' );
		console.error( title, detail );
		let panel = document.getElementById( 'tidewater-runtime-error' );
		if ( ! panel ) {

			panel = document.createElement( 'div' );
			panel.id = 'tidewater-runtime-error';
			panel.setAttribute( 'role', 'alert' );
			panel.style.cssText = 'position:fixed;z-index:2147483647;left:16px;right:16px;bottom:16px;max-height:45vh;overflow:auto;box-sizing:border-box;padding:16px 18px;border:1px solid #e37b7b;border-radius:10px;background:rgba(20,12,16,.97);color:#fff;font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;box-shadow:0 8px 32px #0008;';
			const heading = document.createElement( 'strong' );
			heading.style.cssText = 'display:block;font:600 15px/1.4 system-ui,sans-serif;margin-bottom:8px;color:#ffb4b4';
			const detailEl = document.createElement( 'div' );
			panel.append( heading, detailEl );
			document.body.appendChild( panel );

		}
		panel.children[ 0 ].textContent = title;
		panel.children[ 1 ].textContent = message;

	}

	stop() {

		cancelAnimationFrame( this._raf );

	}

}
