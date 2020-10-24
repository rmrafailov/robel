<!--
 ~ Copyright 2019 The ROBEL Authors.
 ~
 ~ Licensed under the Apache License, Version 2.0 (the "License");
 ~ you may not use this file except in compliance with the License.
 ~ You may obtain a copy of the License at
 ~
 ~     http://www.apache.org/licenses/LICENSE-2.0
 ~
 ~ Unless required by applicable law or agreed to in writing, software
 ~ distributed under the License is distributed on an "AS IS" BASIS,
 ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ~ See the License for the specific language governing permissions and
 ~ limitations under the License.
 -->

<?xml version="1.0"?>
<mujocoinclude name="valve3">
  <body name="valve_base" pos=".0 0.00 0.01" euler="0 0 0" childclass="station">
    <geom class="station_viz_plastic_transparent" type="box" size=".031 .031 .005" pos=".0125 0 -.005"/>
    <geom class="station_viz_metal_grey" mesh="motor" pos="0 0 0.018" euler="0 0 1.57"/>
    <geom class="station_phy_metal" mesh="motor_hull" pos="0 0 0.018" euler="0 0 1.57"/>
    <body name="valve" pos="0 0 0.038" euler="0 0 -1.57">
      <geom class="station_viz_plastic_white" mesh="valve_3" pos="0 0 0"/>
      <!--geom class="station_viz_plastic_red" mesh="valve_3" pos="0 0 0"/-->
      <geom class="station_phy_plastic" type="capsule" pos="0.0 0.038 0.054" size="0.021 0.035" euler="1.57 0 0"/>
      <geom class="station_phy_plastic" type="capsule" pos="-0.034 -0.019 0.054" size="0.021 0.035" euler="1.57 2.0944 0"/>
      <geom class="station_phy_plastic" type="capsule" pos="0.034 -0.019 0.054" size="0.021 0.035" euler="1.57 4.1887 0"/>
      <geom class="station_phy_plastic" type="cylinder" pos="0.0 0.0 0.027" size="0.02 0.0275"/>
      <site name="valve_mark" type="capsule" size=".003 0.0375" pos="0 .0375 .073" rgba="0 0 1 1" euler="1.57 0 0"/>
      <joint name="valve_OBJRx" pos="0 0 0" type="hinge" axis="0 0 1" range="-6.28 6.28" damping=".1" limited="false"/>
    </body>
  </body>
</mujocoinclude>
