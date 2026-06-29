import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from pmt_tasks.asset_config import ASSET_DIR

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2  #  0.003609725* 10 * 2.0 * 3.1415926535*10 * 2.0 * 3.1415926535= 14.2635
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2 # 0.010177520 * 10 * 2.0 * 3.1415926535 * 10 * 2.0 * 3.1415926535 = 40.3051
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2 # 0.025101925 * 10 * 2.0 * 3.1415926535 * 10 * 2.0 * 3.1415926535 = 99.3407
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2 # 0.00425 * 10 * 2.0 * 3.1415926535 * 10 * 2.0 * 3.1415926535 = 16.8495

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ # 2.0 * 2.0 * 0.003609725 * 10 * 2.0 * 3.1415926535 = 0.9065
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ # 2.0 * 2.0 * 0.010177520 * 10 * 2.0 * 3.1415926535 = 2.53
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ # 2.0 * 2.0 * 0.025101925 * 10 * 2.0 * 3.1415926535 = 6.30
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ # 2.0 * 2.0 * 0.00425 * 10 * 2.0 * 3.1415926535 = 1.06825


        # stiffness = {'hip_yaw': 40.3,
        #              'hip_roll': 99.3407,
        #              'hip_pitch': 40.3,
        #              'knee': 99.3407,
        #              'ankle_pitch': 14.2635*2,
        #              'ankle_roll': 14.2635*2,

        #              "waist_roll": 14.2635*2,
        #              "waist_pitch": 14.2635*2,
        #              "waist_yaw": 40.3051,
                     
        #              "shoulder": 14.2635,
        #              "wrist_roll": 14.2635,
        #              "elbow": 14.2635,
                     
        #              "wrist_pitch": 16.8495,
        #              "wtist_yaw": 16.8495,
                     
                     
        #              "hand": 10
                    
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 2.53,
        #              'hip_roll': 6.30,
        #              'hip_pitch': 2.53,
        #              'knee': 6.30,
        #              'ankle_pitch':   0.9065*2,
        #              'ankle_roll':  0.9065*2,


        #              "waist_roll":  0.9065*2,
        #              "waist_pitch":  0.9065*2,
        #              "waist_yaw": 2.53,
                     
        #              "shoulder": 0.9065,
        #              "wrist_roll": 0.9065,
        #              "elbow": 0.9065,
                     
        #              "wrist_pitch": 1.06825,
        #              "wtist_yaw": 1.06825,                    
                     
                    
        #              "hand": 2
        #              }  # [N*m/rad]  # [N*m*s/rad]
        
        
G1_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/unitree_description/urdf/g1/main.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",   #action_scale = 0.25* 88/40.3051=0.545
                ".*_hip_roll_joint", # action_scale = 0.25* 139/99.3407=0.35
                ".*_hip_pitch_joint", # action_scale = 0.25* 88/40.3051=0.545
                ".*_knee_joint", # action_scale = 0.25* 139/99.3407=0.35
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },            
            # effort_limit_sim={
            #     ".*_hip_yaw_joint": 250.0,
            #     ".*_hip_roll_joint": 250.0,
            #     ".*_hip_pitch_joint": 250.0,
            #     ".*_knee_joint": 250.0,
            # },
            # velocity_limit_sim={
            #     ".*_hip_yaw_joint": 100.0,
            #     ".*_hip_roll_joint": 100.0,
            #     ".*_hip_pitch_joint": 100.0,
            #     ".*_knee_joint": 100.0,
            # },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,  
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            # effort_limit_sim={
            #     ".*_shoulder_pitch_joint": 250.0,
            #     ".*_shoulder_roll_joint": 250.0,
            #     ".*_shoulder_yaw_joint": 250.0,
            #     ".*_elbow_joint": 250.0,
            #     ".*_wrist_roll_joint": 250.0,
            #     ".*_wrist_pitch_joint": 50.0,
            #     ".*_wrist_yaw_joint": 50.0,
            # },
            # velocity_limit_sim={
            #     ".*_shoulder_pitch_joint": 150.0,
            #     ".*_shoulder_roll_joint": 150.0,
            #     ".*_shoulder_yaw_joint": 150.0,
            #     ".*_elbow_joint": 150.0,
            #     ".*_wrist_roll_joint": 150.0,
            #     ".*_wrist_pitch_joint": 150.0,
            #     ".*_wrist_yaw_joint": 150.0,
            # },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)
G1_NO_COLLISION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/unitree_description/urdf/g1/main_no_collision/main_no_collision.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.01, 1., 0.01)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",   #action_scale = 0.25* 88/40.3051=0.545
                ".*_hip_roll_joint", # action_scale = 0.25* 139/99.3407=0.35
                ".*_hip_pitch_joint", # action_scale = 0.25* 88/40.3051=0.545
                ".*_knee_joint", # action_scale = 0.25* 139/99.3407=0.35
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },            
            # effort_limit_sim={
            #     ".*_hip_yaw_joint": 250.0,
            #     ".*_hip_roll_joint": 250.0,
            #     ".*_hip_pitch_joint": 250.0,
            #     ".*_knee_joint": 250.0,
            # },
            # velocity_limit_sim={
            #     ".*_hip_yaw_joint": 100.0,
            #     ".*_hip_roll_joint": 100.0,
            #     ".*_hip_pitch_joint": 100.0,
            #     ".*_knee_joint": 100.0,
            # },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,  
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            # effort_limit_sim={
            #     ".*_shoulder_pitch_joint": 250.0,
            #     ".*_shoulder_roll_joint": 250.0,
            #     ".*_shoulder_yaw_joint": 250.0,
            #     ".*_elbow_joint": 250.0,
            #     ".*_wrist_roll_joint": 250.0,
            #     ".*_wrist_pitch_joint": 50.0,
            #     ".*_wrist_yaw_joint": 50.0,
            # },
            # velocity_limit_sim={
            #     ".*_shoulder_pitch_joint": 150.0,
            #     ".*_shoulder_roll_joint": 150.0,
            #     ".*_shoulder_yaw_joint": 150.0,
            #     ".*_elbow_joint": 150.0,
            #     ".*_wrist_roll_joint": 150.0,
            #     ".*_wrist_pitch_joint": 150.0,
            #     ".*_wrist_yaw_joint": 150.0,
            # },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)

G1_ACTION_SCALE = {}
for a in G1_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            G1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
print(G1_ACTION_SCALE)      
print("####G1_ACTION_SCALE####")     